function mha_fwd(
    Q::TileArray4, K::TileArray4, V::TileArray4, O::TileArray4,
    M::TileArray3{Float32}, L::TileArray3{Float32},
    qk_scale::Float32,
    input_pos::Int32,
    H::Int,
    T::Type,
    Dk::Int,
    Dv::Int,
    TILE_M::Int,
    TILE_N::Int,
    QUERY_GROUP_SIZE::Int,
    CAUSAL::Bool,
    EVEN_K::Bool,
)
    padding_mode = ct.PaddingMode.Zero
    i, hb = ct.bid(1), ct.bid(2)
    b, h = fldmod1(hb, H)
    hₖ = cld(h, QUERY_GROUP_SIZE)

    qk_scale_log2 = qk_scale * inv(log(2f0))

    offs_m = reshape((i - 1i32) * TILE_M .+ (ct.arange((TILE_M,), Int32) .- 1i32) .+ input_pos, (1, TILE_M))
    offs_n_tile = reshape(ct.arange((TILE_N,), Int32) .- 1i32, (TILE_N, 1))

    m_i = ct.full((1, TILE_M), -Inf32, Float32)
    l_i = ct.zeros((1, TILE_M), Float32)
    acc = ct.zeros((Dv, TILE_M), Float32)

    q = ct.load(Q, (1, i, h, b), (Dk, TILE_M); padding_mode)

    m_end = input_pos + i * TILE_M
    k_seqlen = size(K, 2)

    if CAUSAL
        mask_start = fld(input_pos + (i - 1i32) * TILE_M, TILE_N)
        mask_start = min(mask_start, fld(k_seqlen, TILE_N))
        kv_tiles = cld(min(Int32(m_end), k_seqlen), TILE_N)
    else
        kv_tiles = cld(k_seqlen, TILE_N)
        mask_start = fld(k_seqlen, TILE_N)
    end

    j = 1i32
    while j <= kv_tiles
        k = ct.load(K, (1, j, hₖ, b), (Dk, TILE_N); padding_mode, latency=2)

        s = muladd(transpose(k) → T, q → T, ct.zeros((TILE_N, TILE_M), Float32))

        if (CAUSAL || !EVEN_K) && j > mask_start
            offs_n = (j - 1i32) * TILE_N .+ offs_n_tile
            mask = ct.full((TILE_N, TILE_M), true, Bool)
            EVEN_K || (mask = mask .& (offs_n .< k_seqlen))
            CAUSAL && (mask = mask .& (offs_m .>= offs_n))
            s = ifelse.(mask, s, -Inf32)
        end

        m_ij = max.(m_i, maximum(s, dims=1) * qk_scale_log2)
        p = exp2.(s .* qk_scale_log2 .- m_ij)
        l_ij = sum(p, dims=1)

        alpha = exp2.(m_i .- m_ij)
        l_i = l_i .* alpha .+ l_ij
        acc = acc .* alpha

        v = ct.load(V, (1, j, hₖ, b), (Dv, TILE_N); padding_mode, latency=4)
        acc = muladd(v → T, p → T, acc)

        m_i = m_ij
        j += 1i32
    end

    o = acc ./ l_i
    ct.store(O, (1, i, h, b), o → eltype(O))
    ct.store(M, (i, h, b), reshape(m_i, (TILE_M,)) .* log(2f0))
    ct.store(L, (i, h, b), reshape(l_i, (TILE_M,)))

    return
end

function mha_bwd_preprocess(
    Ō::TileArray4,
    O::TileArray4,
    Ōscaled::TileArray4,
    L::TileArray3{Float32},
    Δ::TileArray3{Float32},
    H::Int, Dv::Int, TILE_M::Int
)
    padding_mode = ct.PaddingMode.Zero
    i, hb = ct.bid(1), ct.bid(2)
    b, h = cld(hb, H), mod1(hb, H)

    ō = ct.load(Ō, (1, i, h, b), (Dv, TILE_M); padding_mode)
    o  = ct.load(O, (1, i, h, b), (Dv, TILE_M); padding_mode)

    l = reshape(ct.load(L, (i, h, b), (TILE_M,)), (1, TILE_M))

    # optional safety if l can be 0 (masked rows / OOB):
    inv_l = ifelse.(l .== 0f0, 0f0, 1f0 ./ l)

    ōs = ō .* inv_l # TODO just divide by l?
    ct.store(Ōscaled, (1, i, h, b), ōs)

    δ = sum(ōs .* o, dims=1)
    ct.store(Δ, (i, h, b), reshape(δ, (TILE_M,)))

    return
end

function mha_bwd(
    Q::TileArray4, K::TileArray4, V::TileArray4,
    Ōscaled::TileArray4,
    M::TileArray3{Float32},
    Δ::TileArray3{Float32},
    Q̄::TileArray4, K̄::TileArray4, V̄::TileArray4,
    qk_scale::Float32,
    input_pos::Integer,
    H::Integer,
    T::Type,
    Dk::Int,
    Dv::Int,
    TILE_M::Int,
    TILE_N::Int,
    QUERY_GROUP_SIZE::Int,
    CAUSAL::Bool,
    EVEN_K::Bool,
)
    padding_mode = ct.PaddingMode.Zero
    hb = ct.bid(1)
    b, h = fldmod1(hb, H)
    hₖ = cld(h, QUERY_GROUP_SIZE)

    q_seqlen, k_seqlen = size(Q, 2), size(K, 2)
    q_tiles = cld(q_seqlen, TILE_M)
    kv_tiles = cld(k_seqlen, TILE_N)

    offs_n_base = reshape(ct.arange((TILE_N,), Int32) .- 1i32, (TILE_N, 1))

    j = 1i32
    while j <= kv_tiles
        k = ct.load(K, (1, j, hₖ, b), (Dk, TILE_N); padding_mode)
        v = ct.load(V, (1, j, hₖ, b), (Dv, TILE_N); padding_mode)

        k̄_acc = ct.zeros((Dk, TILE_N), Float32)
        v̄_acc = ct.zeros((Dv, TILE_N), Float32)

        offs_n = (j - 1i32) * TILE_N .+ offs_n_base

        i = 1i32
        while i <= q_tiles
            q = ct.load(Q, (1, i, h, b), (Dk, TILE_M); padding_mode, allow_tma=false)
            ō = ct.load(Ōscaled, (1, i, h, b), (Dv, TILE_M); padding_mode, allow_tma=false)

            m = reshape(ct.load(M, (i, h, b), (TILE_M,), latency=1), (1, TILE_M))
            δ = reshape(ct.load(Δ, (i, h, b), (TILE_M,), latency=1), (1, TILE_M))

            s = muladd((k)ᵀ → T, q → T, ct.zeros((TILE_N, TILE_M), Float32))
            s = s .* qk_scale

            if CAUSAL || !EVEN_K
                offs_m = reshape((i - 1i32) * TILE_M .+ (ct.arange((TILE_M,), Int32) .- 1i32) .+ input_pos, (1, TILE_M))
                mask = ct.full((TILE_N, TILE_M), true, Bool)
                EVEN_K || (mask = mask .& (offs_n .< k_seqlen))
                CAUSAL && (mask = mask .& (offs_m .>= offs_n))
                s = ifelse.(mask, s, -Inf32, Float32)
            end

            p = exp.(s .- m)
            v̄_acc = muladd(ō → T, (p)ᵀ → T, v̄_acc)

            p̄ = muladd((v)ᵀ → T, ō → T, ct.zeros((TILE_N, TILE_M), Float32))

            s̄ = (p .* (p̄ .- δ)) .* qk_scale

            q̄ = ct.load(Q̄, (1, i, h, b), (Dk, TILE_M), allow_tma=false)
            q̄ = muladd(k → T, s̄ → T, q̄)
            ct.store(Q̄, (1, i, h, b), q̄)

            k̄_acc = muladd(q → T, (s̄)ᵀ → T, k̄_acc)

            i += 1i32
        end

        store = isone(QUERY_GROUP_SIZE) ? ct.store : ct.atomic_add
        store(K̄, (1, j, hₖ, b), k̄_acc → eltype(K̄))
        store(V̄, (1, j, hₖ, b), v̄_acc → eltype(V̄))

        j += 1i32
    end
    
    return
end

function flash_attention(
    Q, K, V; causal,
    compute = eltype(Q),
    verify = nothing
)
    Dq, SeqLen_Q, Heads, Batch = size(Q)
    Dk, SeqLen_K, Heads_KV, Batch_K = size(K)
    Dv, SeqLen_V, Heads_V, Batch_V = size(V)
    @assert Dq == Dk
    @assert SeqLen_K == SeqLen_V
    @assert Heads_KV == Heads_V
    @assert iszero(Heads % Heads_KV)

    O = similar(Q, Dv, SeqLen_Q, Heads, Batch)
    M = similar(Q, Float32, SeqLen_Q, Heads, Batch)
    L = similar(Q, Float32, SeqLen_Q, Heads, Batch)

    query_group_size = Heads ÷ Heads_KV
    qk_scale = Float32(1 / sqrt(Dk))
    input_pos = Int32(0)
    Dk_pow2 = nextpow(2, Dk)
    Dv_pow2 = nextpow(2, Dv)

    key = (eltype(Q), compute, Dk_pow2, Dv_pow2)

    autotune_launch(mha_fwd,
        CartesianSpace(TILE_M=(32, 64, 128), TILE_N=(32, 64, 128), occupancy=(1, 2, 4)),
        cfg -> (cld(SeqLen_Q, cfg.TILE_M), Heads * Batch),
        cfg -> (
            Q, K, V, O, M, L,
            qk_scale, input_pos, Heads,
            Constant(compute),
            Constant(Dk_pow2),
            Constant(Dv_pow2),
            Constant(cfg.TILE_M),
            Constant(cfg.TILE_N),
            Constant(query_group_size),
            Constant(causal),
            Constant(iszero(SeqLen_K % cfg.TILE_N))
        );
        key, verify
    )

    return O, M, L
end

function ∇flash_attention(
    Ō, Q, K, V, O, M, L; causal,
    compute = eltype(Q),
    verify = nothing,
)
    Dk, SeqLen_Q, H, B = size(Q)
    Dk_K, SeqLen_K, H_KV, B_K = size(K)
    Dv, SeqLen_V, H_V, B_V = size(V)
    @assert Dk == Dk_K
    @assert SeqLen_K == SeqLen_V
    @assert H_KV == H_V
    @assert B == B_K == B_V
    @assert size(O, 1) == Dv
    @assert size(Ō, 1) == Dv
    @assert iszero(H % H_KV)
    
    query_group_size = H ÷ H_KV
    qk_scale = Float32(1 / sqrt(Dk))
    input_pos = Int32(0)

    Ōscaled, Δ = similar(Ō), similar(M)
    Q̄, K̄, V̄ = similar.((Q, K, V))
    isone(query_group_size) || fill!.((Q̄, K̄, V̄), 0)

    ct.launch(mha_bwd_preprocess,
        (cld(SeqLen_Q, 32), H * B),
        Ō, O, Ōscaled, L, Δ,
        H, Constant(Dv), Constant(32)
    )

    key = (eltype(Q), compute, Dk_pow2, Dv_pow2)

    autotune_launch(mha_bwd,
        CartesianSpace(TILE_M=(32, 64, 128), TILE_N=(32, 64, 128), occupancy=(1, 2, 4)),
        cfg -> H * B,
        cfg -> (
            Q, K, V, Ōscaled, M, Δ,
            Q̄, K̄, V̄,
            qk_scale, input_pos, H,
            Constant(compute),
            Constant(nextpow(2, Dk)),
            Constant(nextpow(2, Dv)),
            Constant(cfg.TILE_M),
            Constant(cfg.TILE_N),
            Constant(query_group_size),
            Constant(causal),
            Constant(iszero(SeqLen_K % TILE_N))
        );
        key, verify
    )

    return Q̄, K̄, V̄
end

#=
verify = () -> let
    ref = Primitives.attention(Q, K, V; causal)
    () -> (isapprox(O, ref, rtol=1e-2))
end
=#
