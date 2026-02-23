function mha_fwd(
    Q::TileArray4, K::TileArray4, V::TileArray4, O::TileArray4,
    M::Optional{TileArray3{Float32}}, L::Optional{TileArray3{Float32}},
    B::Optional{TileArray4},
    k_lengths::TileVector{Int32},
    q_lengths::TileVector{Int32},
    qk_scale::Float32,
    input_pos::Int32,
    H::Int,
    Tc::Type, Tacc::Type,
    Dk::Int, Dv::Int,
    TILE_M::Int, TILE_N::Int,
    QUERY_GROUP_SIZE::Int,
    CAUSAL::Bool,
    BIAS_HEADS::Int,
    BIAS_BATCH::Int,
)
    padding_mode = ct.PaddingMode.Zero
    i, hb = ct.bid(1), ct.bid(2)
    b, h = fldmod1(hb, H)
    hₖ = cld(h, QUERY_GROUP_SIZE)

    q_len = q_lengths[b]
    (i - 1i32) * TILE_M >= q_len && return

    offs_m = reshape((i - 1i32) * TILE_M .+ (ct.arange((TILE_M,), Int32) .- 1i32) .+ input_pos, (1, TILE_M))
    offs_n_tile = reshape(ct.arange((TILE_N,), Int32) .- 1i32, (TILE_N, 1))

    m_i = ct.full((1, TILE_M), -Inf32, Float32)
    l_i = ct.zeros((1, TILE_M), Float32)
    acc = ct.zeros((Dv, TILE_M), Tacc)

    q = ct.load(Q, (1, i, h, b), (Dk, TILE_M); padding_mode)

    k_len = k_lengths[b]
    m_end = input_pos + i * TILE_M

    if CAUSAL
        mask_start = min(fld(input_pos + (i - 1i32) * TILE_M, TILE_N), fld(k_len, TILE_N))
        kv_tiles = cld(min(Int32(m_end), k_len), TILE_N)
    else
        mask_start = fld(k_len, TILE_N)
        kv_tiles = cld(k_len, TILE_N)
    end

    if B isa TileArray
        hᵇ = mod1(h, BIAS_HEADS)
        bᵇ = mod1(b, BIAS_BATCH)
    end

    j = 1i32
    while j <= kv_tiles
        k = ct.load(K, (1, j, hₖ, b), (Dk, TILE_N); padding_mode, latency=2)

        s = muladd((k)ᵀ → Tc, q → Tc, ct.zeros((TILE_N, TILE_M), Tacc))
        s = s * Tacc(qk_scale)

        if B isa TileArray
            bias = ct.load(B, (j, i, hᵇ, bᵇ), (TILE_N, TILE_M))
            s = s .+ bias → Tacc
        end

        if j > mask_start
            offs_n = (j - 1i32) * TILE_N .+ offs_n_tile
            mask = offs_n .< k_len
            CAUSAL && (mask = mask .& (offs_m .>= offs_n))
            s = ifelse.(mask, s, Tacc(-Inf32))
        end

        m_ij = max.(m_i, maximum(s, dims=1))
        p = exp.(s .- m_ij)
        l_ij = sum(p, dims=1)
        alpha = exp.(m_i .- m_ij)
        l_i = l_i .* alpha .+ l_ij
        acc = acc .* Tacc.(alpha)

        v = ct.load(V, (1, j, hₖ, b), (Dv, TILE_N); padding_mode, latency=4)
        acc = muladd(v → Tc, p → Tc, acc)

        m_i = m_ij
        j += 1i32
    end

    o = acc ./ l_i
    ct.store(O, (1, i, h, b), o → eltype(O))
    isnothing(M) || ct.store(M, (i, h, b), reshape(m_i, (TILE_M,)))
    isnothing(L) || ct.store(L, (i, h, b), reshape(l_i, (TILE_M,)))

    return
end

function mha_bwd_preprocess(
    Ō::TileArray4,
    O::TileArray4,
    Ō′::TileArray4,
    L::TileArray3{Float32},
    Δ::TileArray3{Float32},
    q_lengths::TileVector{Int32},
    H::Int, Dv::Int, TILE_M::Int
)
    padding_mode = ct.PaddingMode.Zero
    i, hb = ct.bid(1), ct.bid(2)
    b, h = cld(hb, H), mod1(hb, H)

    q_len = q_lengths[b]
    (i - 1i32) * TILE_M >= q_len && return

    ō = ct.load(Ō, (1, i, h, b), (Dv, TILE_M); padding_mode)
    o  = ct.load(O, (1, i, h, b), (Dv, TILE_M); padding_mode)

    l = reshape(ct.load(L, (i, h, b), (TILE_M,)), (1, TILE_M))

    ō′ = ō .* ifelse.(l .== 0f0, 0f0, 1f0 ./ l)
    ct.store(Ō′, (1, i, h, b), ō′ → eltype(Ō′))

    δ = sum(ō′ .* o, dims=1)
    ct.store(Δ, (i, h, b), reshape(δ, (TILE_M,)))

    return
end

function mha_bwd(
    Q::TileArray4, K::TileArray4, V::TileArray4,
    Ō′::TileArray4,
    M::TileArray3{Float32},
    Δ::TileArray3{Float32},
    Q̄::TileArray4, K̄::TileArray4, V̄::TileArray4,
    B::Optional{TileArray4},
    B̄::Optional{TileArray4},
    k_lengths::TileVector{Int32},
    q_lengths::TileVector{Int32},
    qk_scale::Float32,
    input_pos::Integer,
    H::Integer,
    Tc::Type, Tacc::Type,
    Dk::Int, Dv::Int,
    TILE_M::Int, TILE_N::Int,
    QUERY_GROUP_SIZE::Int,
    CAUSAL::Bool,
    BIAS_HEADS::Int,
    BIAS_BATCH::Int,
    BIAS_ATOMIC::Bool,
)
    padding_mode = ct.PaddingMode.Zero
    hb = ct.bid(1)
    b, h = fldmod1(hb, H)
    hₖ = cld(h, QUERY_GROUP_SIZE)

    k_len = k_lengths[b]
    q_len = q_lengths[b]

    q_tiles = cld(q_len, TILE_M)
    kv_tiles = cld(k_len, TILE_N)

    offs_n_base = reshape(ct.arange((TILE_N,), Int32) .- 1i32, (TILE_N, 1))

    if B isa TileArray
        hᵇ = mod1(h, BIAS_HEADS)
        bᵇ = mod1(b, BIAS_BATCH)
    end

    j = 1i32
    while j <= kv_tiles
        k = ct.load(K, (1, j, hₖ, b), (Dk, TILE_N); padding_mode)
        v = ct.load(V, (1, j, hₖ, b), (Dv, TILE_N); padding_mode)

        k̄_acc = ct.zeros((Dk, TILE_N), Tacc)
        v̄_acc = ct.zeros((Dv, TILE_N), Tacc)

        offs_n = (j - 1i32) * TILE_N .+ offs_n_base
        pad_mask_needed = j > fld(k_len, TILE_N)

        i = 1i32
        while i <= q_tiles
            q = ct.load(Q, (1, i, h, b), (Dk, TILE_M); padding_mode, allow_tma=false)
            ō = ct.load(Ō′, (1, i, h, b), (Dv, TILE_M); padding_mode, allow_tma=false)

            m = reshape(ct.load(M, (i, h, b), (TILE_M,), latency=1), (1, TILE_M))
            δ = reshape(ct.load(Δ, (i, h, b), (TILE_M,), latency=1), (1, TILE_M))

            s = muladd((k)ᵀ → Tc, q → Tc, ct.zeros((TILE_N, TILE_M), Tacc))
            s = s * Tacc(qk_scale)

            if B isa TileArray
                pair = ct.load(B, (j, i, hᵇ, bᵇ), (TILE_N, TILE_M))
                s = s .+ pair → Tacc
            end

            if CAUSAL || pad_mask_needed
                offs_m = reshape((i - 1i32) * TILE_M .+ (ct.arange((TILE_M,), Int32) .- 1i32) .+ input_pos, (1, TILE_M))
                mask = offs_n .< k_len
                CAUSAL && (mask = mask .& (offs_m .>= offs_n))
                s = ifelse.(mask, s, Tacc(-Inf32))
            end

            p = exp.(s .- Tacc.(m))
            v̄_acc = muladd(ō → Tc, (p)ᵀ → Tc, v̄_acc)

            p̄ = muladd((v)ᵀ → Tc, ō → Tc, ct.zeros((TILE_N, TILE_M), Tacc))

            ds = p .* (p̄ .- Tacc.(δ))

            if B̄ isa TileArray
                bias_store = BIAS_ATOMIC ? ct.atomic_add : ct.store
                bias_store(B̄, (j, i, hᵇ, bᵇ), ds → eltype(B̄))
            end

            s̄ = ds * Tacc(qk_scale)

            q̄ = ct.load(Q̄, (1, i, h, b), (Dk, TILE_M), allow_tma=false)
            q̄ = muladd(k → Tc, s̄ → Tc, q̄ → Tacc)
            ct.store(Q̄, (1, i, h, b), q̄ → eltype(Q̄))

            k̄_acc = muladd(q → Tc, (s̄)ᵀ → Tc, k̄_acc)

            i += 1i32
        end

        store = isone(QUERY_GROUP_SIZE) ? ct.store : ct.atomic_add
        store(K̄, (1, j, hₖ, b), k̄_acc → eltype(K̄))
        store(V̄, (1, j, hₖ, b), v̄_acc → eltype(V̄))

        j += 1i32
    end

    return
end

function flash_attention!(O,
    Q, K, V, B;
    causal,
    k_lengths = nothing,
    q_lengths = nothing,
    tensorcore = tensorcore_type(eltype(Q)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing
)
    Dq, SeqLen_Q, Heads, Batch = size(Q)
    Dk, SeqLen_K, Heads_KV, Batch_K = size(K)
    Dv, SeqLen_V, Heads_V, Batch_V = size(V)
    @assert Dq == Dk
    @assert SeqLen_K == SeqLen_V
    @assert Heads_KV == Heads_V
    @assert iszero(Heads % Heads_KV)

    M = similar(Q, Float32, SeqLen_Q, Heads, Batch)
    L = similar(Q, Float32, SeqLen_Q, Heads, Batch)

    if isnothing(k_lengths)
        k_lengths = fill!(similar(Q, Int32, Batch), SeqLen_K)
    end
    if isnothing(q_lengths)
        q_lengths = fill!(similar(Q, Int32, Batch), SeqLen_Q)
    end

    query_group_size = Heads ÷ Heads_KV
    qk_scale = Float32(1 / sqrt(Dk))
    input_pos = Int32(0)
    Dk_pow2 = nextpow(2, Dk)
    Dv_pow2 = nextpow(2, Dv)

    bias_heads = isnothing(B) ? 0 : size(B, 3)
    bias_batch = isnothing(B) ? 0 : size(B, 4)

    key = (eltype(Q), tensorcore, accumulate,
        Dk_pow2, Dv_pow2, nextpow.(4, (SeqLen_Q, SeqLen_K)),
        !isnothing(B))

    autotune_launch(mha_fwd,
        CartesianSpace(
            TILE_M=(32, 64, 128, 256), TILE_N=(32, 64, 128, 256),
            occupancy=(1, 2, 4, 8)
        ),
        cfg -> (cld(SeqLen_Q, cfg.TILE_M), Heads * Batch),
        cfg -> (
            Q, K, V, O, M, L, B, k_lengths, q_lengths,
            qk_scale, input_pos, Heads,
            Constant(tensorcore), Constant(accumulate),
            Constant(Dk_pow2),
            Constant(Dv_pow2),
            Constant(cfg.TILE_M),
            Constant(cfg.TILE_N),
            Constant(query_group_size),
            Constant(causal),
            Constant(bias_heads),
            Constant(bias_batch),
        );
        key, verify
    )

    return M, L
end

function ∇flash_attention!(
    Q̄, K̄, V̄, B̄, Ō,
    Q, K, V, B, O,
    M, L;
    causal,
    k_lengths = nothing,
    q_lengths = nothing,
    tensorcore = tensorcore_type(eltype(Q)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing,
)
    Dq, SeqLen_Q, Heads, Batch = size(Q)
    Dk, SeqLen_K, Heads_KV, Batch_K = size(K)
    Dv, SeqLen_V, Heads_V, Batch_V = size(V)
    @assert Dq == Dk
    @assert SeqLen_K == SeqLen_V
    @assert Heads_KV == Heads_V
    @assert Batch == Batch_K == Batch_V
    @assert size(O, 1) == Dv
    @assert size(Ō, 1) == Dv
    @assert iszero(Heads % Heads_KV)

    k_lengths = @something k_lengths fill!(similar(Q, Int32, Batch), SeqLen_K)
    q_lengths = @something q_lengths fill!(similar(Q, Int32, Batch), SeqLen_Q)

    query_group_size = Heads ÷ Heads_KV
    qk_scale = Float32(1 / sqrt(Dk))
    input_pos = Int32(0)
    Dk_pow2 = nextpow(2, Dk)
    Dv_pow2 = nextpow(2, Dv)

    bias_heads = isnothing(B) ? 0 : size(B, 3)
    bias_batch = isnothing(B) ? 0 : size(B, 4)
    bias_atomic = !isnothing(B) && (bias_heads < Heads || bias_batch < Batch)

    Ō′, Δ = similar(Ō), similar(M)

    ct.launch(mha_bwd_preprocess,
        (cld(SeqLen_Q, 32), Heads * Batch),
        Ō, O, Ō′, L, Δ, q_lengths,
        Heads, Constant(Dv), Constant(32)
    )

    key = (eltype(Q), tensorcore, accumulate,
        Dk_pow2, Dv_pow2, nextpow.(4, (SeqLen_Q, SeqLen_K)),
        !isnothing(B))

    autotune_launch(mha_bwd,
        CartesianSpace(
            TILE_M=(32, 64, 128, 256), TILE_N=(32, 64, 128, 256),
            occupancy=(1, 2, 4, 8)
        ),
        cfg -> Heads * Batch,
        cfg -> (
            Q, K, V, Ō′, M, Δ,
            fill!.((Q̄, K̄, V̄), 0)...,
            B,
            isnothing(B̄) ? B̄ : fill!(B̄, 0),
            k_lengths, q_lengths,
            qk_scale, input_pos, Heads,
            Constant(tensorcore), Constant(accumulate),
            Constant(nextpow(2, Dk)),
            Constant(nextpow(2, Dv)),
            Constant(cfg.TILE_M),
            Constant(cfg.TILE_N),
            Constant(query_group_size),
            Constant(causal),
            Constant(bias_heads),
            Constant(bias_batch),
            Constant(bias_atomic),
        );
        key, verify
    )

    return nothing
end
