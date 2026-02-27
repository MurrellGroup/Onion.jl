# K, U: (I, D, H) — gate/up weights in (d_inter, d_head, n_heads) layout
#                     I = E * D_E when using experts (flattened along I dim)
# V:    (D, I, H) — down weight in (d_head, d_inter, n_heads) layout
# Q, O: (D, H, L) — input/output in (d_head, n_heads, seq) layout
# R:    (E, H, L) — optional router weights (nothing for non-expert mode)
# R̄:   (E, H, L) — optional router gradient output (dQ kernel only)

function mhffn_fwd(
    Q::TileArray3, K::TileArray3, U::TileArray3, V::TileArray3,
    O::TileArray3,
    R::Optional{TileArray3},
    Tc::Type, Tacc::Type,
    TILE_D::Int, TILE_L::Int, TILE_I::Int,
    D_E::Optional{Int},
)
    padding_mode = ct.PaddingMode.Zero
    j, h = ct.bid(1), ct.bid(2)

    q = dropdims(ct.load(Q, (1, h, j), (TILE_D, 1, TILE_L); padding_mode), dims=2)

    acc = ct.zeros((TILE_D, TILE_L), Tacc)

    num_i = cld(size(K, 1), TILE_I)

    if R isa TileArray
        tiles_per_expert = D_E ÷ TILE_I
    end

    i = 1i32
    while i <= num_i
        k = ct.load(K, (i, 1, h), (TILE_I, TILE_D); padding_mode)
        u = ct.load(U, (i, 1, h), (TILE_I, TILE_D); padding_mode)
        v = ct.load(V, (1, i, h), (TILE_D, TILE_I); padding_mode)

        m = muladd(k → Tc, q → Tc, ct.zeros((TILE_I, TILE_L), Tacc))
        n = muladd(u → Tc, q → Tc, ct.zeros((TILE_I, TILE_L), Tacc))

        a = m ./ (1 .+ exp.(0 .- m)) .* n

        if R isa TileArray
            e = fld(i - 1i32, tiles_per_expert) + 1i32
            r = dropdims(ct.load(R, (e, h, j), (1, 1, TILE_L); padding_mode), dims=1)
            a = a .* r
        end

        acc = muladd(v → Tc, a → Tc, acc)

        i += 1i32
    end

    ct.store(O, (1, h, j), reshape(acc, (TILE_D, 1, TILE_L)) → eltype(O))

    return
end

function mhffn_bwd_dq(
    Q::TileArray3, K::TileArray3, U::TileArray3, V::TileArray3,
    Ō::TileArray3, Q̄::TileArray3,
    R::Optional{TileArray3}, R̄::Optional{TileArray3},
    Tc::Type, Tacc::Type,
    TILE_D::Int, TILE_L::Int, TILE_I::Int,
    num_i::Int,
    D_E::Optional{Int},
)
    padding_mode = ct.PaddingMode.Zero
    j, h = ct.bid(1), ct.bid(2)

    q = dropdims(ct.load(Q, (1, h, j), (TILE_D, 1, TILE_L); padding_mode), dims=2)
    ō = dropdims(ct.load(Ō, (1, h, j), (TILE_D, 1, TILE_L); padding_mode), dims=2)

    q̄_acc = ct.zeros((TILE_D, TILE_L), Tacc)

    if R isa TileArray
        tiles_per_expert = D_E ÷ TILE_I
        dr_acc = ct.zeros((1, TILE_L), Tacc)
    end

    i = 1i32
    while i <= num_i
        k = ct.load(K, (i, 1, h), (TILE_I, TILE_D); padding_mode)
        u = ct.load(U, (i, 1, h), (TILE_I, TILE_D); padding_mode)
        v = ct.load(V, (1, i, h), (TILE_D, TILE_I); padding_mode)

        m = muladd(k → Tc, q → Tc, ct.zeros((TILE_I, TILE_L), Tacc))
        n = muladd(u → Tc, q → Tc, ct.zeros((TILE_I, TILE_L), Tacc))

        sig = 1 ./ (1 .+ exp.(0 .- m))
        silu_m = m .* sig

        ā = muladd((v)ᵀ → Tc, ō → Tc, ct.zeros((TILE_I, TILE_L), Tacc))

        dsilu_dm = sig .* (1 .+ m .* (1 .- sig))

        if R isa TileArray
            e = fld(i - 1i32, tiles_per_expert) + 1i32
            r = dropdims(ct.load(R, (e, h, j), (1, 1, TILE_L); padding_mode), dims=1)

            # dR_e = Σ_tiles <ā, silu(m) ⊙ n> summed over intermediate dim
            dr_acc = dr_acc .+ sum(ā .* silu_m .* n, dims=1)

            # Flush at expert boundary (TILE_I divides D_E, so this is exact)
            if iszero(mod(i, tiles_per_expert))
                ct.store(R̄, (e, h, j), reshape(dr_acc, (1, 1, TILE_L)) → eltype(R̄))
                dr_acc = ct.zeros((1, TILE_L), Tacc)
            end

            M̄ = ā .* n .* dsilu_dm .* r
            N̄ = ā .* silu_m .* r
        else
            M̄ = ā .* n .* dsilu_dm
            N̄ = ā .* silu_m
        end

        q̄_acc = muladd((k)ᵀ → Tc, M̄ → Tc, q̄_acc)
        q̄_acc = muladd((u)ᵀ → Tc, N̄ → Tc, q̄_acc)

        i += 1i32
    end

    ct.store(Q̄, (1, h, j), reshape(q̄_acc, (TILE_D, 1, TILE_L)) → eltype(Q̄))

    return
end

function mhffn_bwd_dkuv(
    Q::TileArray3, K::TileArray3, U::TileArray3, V::TileArray3,
    Ō::TileArray3, K̄::TileArray3, Ū::TileArray3, V̄::TileArray3,
    R::Optional{TileArray3},
    Tc::Type, Tacc::Type,
    TILE_D::Int, TILE_L::Int, TILE_I::Int,
    D_E::Optional{Int},
)
    padding_mode = ct.PaddingMode.Zero
    i, h = ct.bid(1), ct.bid(2)

    k = ct.load(K, (i, 1, h), (TILE_I, TILE_D); padding_mode)
    u = ct.load(U, (i, 1, h), (TILE_I, TILE_D); padding_mode)
    v = ct.load(V, (1, i, h), (TILE_D, TILE_I); padding_mode)

    k̄_acc = ct.zeros((TILE_I, TILE_D), Tacc)
    ū_acc = ct.zeros((TILE_I, TILE_D), Tacc)
    v̄_acc = ct.zeros((TILE_D, TILE_I), Tacc)

    if R isa TileArray
        tiles_per_expert = D_E ÷ TILE_I
        e = fld(i - 1i32, tiles_per_expert) + 1i32
    end

    num_j = cld(size(Q, 3), TILE_L)
    j = 1i32
    while j <= num_j
        q = dropdims(ct.load(Q, (1, h, j), (TILE_D, 1, TILE_L); padding_mode), dims=2)
        ō = dropdims(ct.load(Ō, (1, h, j), (TILE_D, 1, TILE_L); padding_mode), dims=2)

        m = muladd(k → Tc, q → Tc, ct.zeros((TILE_I, TILE_L), Tacc))
        n = muladd(u → Tc, q → Tc, ct.zeros((TILE_I, TILE_L), Tacc))

        sig = 1 ./ (1 .+ exp.(0 .- m))
        silu_m = m .* sig

        a = silu_m .* n

        ā = muladd((v)ᵀ → Tc, ō → Tc, ct.zeros((TILE_I, TILE_L), Tacc))

        dsilu_dm = sig .* (1 .+ m .* (1 .- sig))

        if R isa TileArray
            r = dropdims(ct.load(R, (e, h, j), (1, 1, TILE_L); padding_mode), dims=1)
            a = a .* r
            M̄ = ā .* n .* dsilu_dm .* r
            N̄ = ā .* silu_m .* r
        else
            M̄ = ā .* n .* dsilu_dm
            N̄ = ā .* silu_m
        end

        k̄_acc = muladd(M̄ → Tc, (q)ᵀ → Tc, k̄_acc)
        ū_acc = muladd(N̄ → Tc, (q)ᵀ → Tc, ū_acc)
        v̄_acc = muladd(ō → Tc, (a)ᵀ → Tc, v̄_acc)

        j += 1i32
    end

    ct.store(K̄, (i, 1, h), k̄_acc → eltype(K̄))
    ct.store(Ū, (i, 1, h), ū_acc → eltype(Ū))
    ct.store(V̄, (1, i, h), v̄_acc → eltype(V̄))

    return
end

# ── Launchers ─────────────────────────────────────────────────────────

function multihead_ffn!(O,
    Q, K, U, V;
    R = nothing,
    tensorcore = tensorcore_type(eltype(Q)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing
)
    D, H, L = size(Q)
    I = size(K, 1)

    @assert size(K) == (I, D, H) "K must be (d_inter, d_head, n_heads), got $(size(K))"
    @assert size(U) == (I, D, H) "U must be (d_inter, d_head, n_heads), got $(size(U))"
    @assert size(V) == (D, I, H) "V must be (d_head, d_inter, n_heads), got $(size(V))"
    @assert size(O) == (D, H, L) "O must match Q size $(size(Q)), got $(size(O))"

    if !isnothing(R)
        E = size(R, 1)
        D_E = I ÷ E
        @assert size(R) == (E, H, L) "R must be (n_experts, n_heads, seq), got $(size(R))"
        tile_i_opts = Tuple(filter(t -> iszero(D_E % t), (32, 64, 128)))
        @assert !isempty(tile_i_opts) "D_E=$D_E must be divisible by at least one of (32, 64, 128)"
    else
        tile_i_opts = (32, 64, 128)
        D_E = nothing
    end

    key = (eltype(Q), tensorcore, accumulate, D, !isnothing(R))

    autotune_launch(mhffn_fwd,
        CartesianSpace(TILE_L=(32, 64, 128), TILE_I=tile_i_opts, occupancy=(1, 2, 4)),
        cfg -> (cld(L, cfg.TILE_L), H),
        cfg -> (
            Q, K, U, V, O, R,
            Constant(tensorcore), Constant(accumulate),
            Constant(D), Constant(cfg.TILE_L), Constant(cfg.TILE_I),
            D_E,
        );
        key, verify
    )

    return ()
end

function ∇multihead_ffn!(Q̄, K̄, Ū, V̄,
    Ō, Q, K, U, V;
    R = nothing,
    R̄ = nothing,
    D_E = 0,
    tensorcore = tensorcore_type(eltype(Q)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing
)
    D, H, L = size(Q)
    I = size(K, 1)

    @assert size(K) == (I, D, H) "K must be (d_inter, d_head, n_heads), got $(size(K))"
    @assert size(U) == (I, D, H) "U must be (d_inter, d_head, n_heads), got $(size(U))"
    @assert size(V) == (D, I, H) "V must be (d_head, d_inter, n_heads), got $(size(V))"
    @assert size(Ō) == (D, H, L) "Ō must match Q size $(size(Q)), got $(size(Ō))"

    if !isnothing(R)
        E = size(R, 1)
        @assert !isnothing(R̄) "R̄ must be provided when R is provided"
        @assert size(R) == (E, H, L)
        @assert size(R̄) == (E, H, L)
        @assert D_E > 0 && E * D_E == I
        tile_i_opts = Tuple(filter(t -> iszero(D_E % t), (32, 64, 128)))
        @assert !isempty(tile_i_opts)
    else
        tile_i_opts = (32, 64, 128)
    end

    key = (eltype(Q), tensorcore, accumulate, D, !isnothing(R))

    autotune_launch(mhffn_bwd_dq,
        CartesianSpace(TILE_L=(32, 64, 128), TILE_I=tile_i_opts, occupancy=(1, 2, 4)),
        cfg -> (cld(L, cfg.TILE_L), H),
        cfg -> (
            Q, K, U, V, Ō, Q̄,
            R, R̄,
            Constant(tensorcore), Constant(accumulate),
            Constant(D), Constant(cfg.TILE_L), Constant(cfg.TILE_I),
            Constant(cld(I, cfg.TILE_I)),
            D_E,
        );
        key, verify
    )

    autotune_launch(mhffn_bwd_dkuv,
        CartesianSpace(TILE_L=(32, 64, 128), TILE_I=tile_i_opts, occupancy=(1, 2, 4)),
        cfg -> (cld(I, cfg.TILE_I), H),
        cfg -> (
            Q, K, U, V, Ō, K̄, Ū, V̄,
            R,
            Constant(tensorcore), Constant(accumulate),
            Constant(D), Constant(cfg.TILE_L), Constant(cfg.TILE_I),
            D_E,
        );
        key, verify
    )

    return nothing
end
