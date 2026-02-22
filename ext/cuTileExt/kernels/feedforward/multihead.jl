function mhffn_fwd(
    Q::TileArray3, K::TileArray3, U::TileArray3, V::TileArray3,
    O::TileArray3,
    T::Type,
    TILE_D::Int, TILE_L::Int, TILE_I::Int
)
    padding_mode = ct.PaddingMode.Zero
    j, h = ct.bid(1), ct.bid(2)

    q = dropdims(ct.load(Q, (1, h, j), (TILE_D, 1, TILE_L); padding_mode), dims=2)

    acc = ct.zeros((TILE_D, TILE_L), Float32)

    num_i = cld(size(K, 2), TILE_I)
    i = 1i32
    while i <= num_i
        k = ct.load(K, (1, i, h), (TILE_D, TILE_I); padding_mode)
        u = ct.load(U, (1, i, h), (TILE_D, TILE_I); padding_mode)
        v = ct.load(V, (1, i, h), (TILE_D, TILE_I); padding_mode)

        m = muladd((k)ᵀ → T, q → T, ct.zeros((TILE_I, TILE_L), Float32))
        n = muladd((u)ᵀ → T, q → T, ct.zeros((TILE_I, TILE_L), Float32))

        a = m ./ (1 .+ exp.(0 .- m)) .* n
        acc = muladd(v → T, a → T, acc)

        i += 1i32
    end

    ct.store(O, (1, h, j), reshape(acc, (TILE_D, 1, TILE_L)) → eltype(O))

    return
end

function mhffn_bwd_dq(
    Q::TileArray3, K::TileArray3, U::TileArray3, V::TileArray3,
    Ō::TileArray3, Q̄::TileArray3,
    T::Type,
    TILE_D::Int, TILE_L::Int, TILE_I::Int,
    num_i::Int
)
    padding_mode = ct.PaddingMode.Zero
    j, h = ct.bid(1), ct.bid(2)

    q = dropdims(ct.load(Q, (1, h, j), (TILE_D, 1, TILE_L); padding_mode), dims=2)
    ō = dropdims(ct.load(Ō, (1, h, j), (TILE_D, 1, TILE_L); padding_mode), dims=2)

    q̄_acc = ct.zeros((TILE_D, TILE_L), Float32)

    i = 1i32
    while i <= num_i
        k = ct.load(K, (1, i, h), (TILE_D, TILE_I); padding_mode)
        u = ct.load(U, (1, i, h), (TILE_D, TILE_I); padding_mode)
        v = ct.load(V, (1, i, h), (TILE_D, TILE_I); padding_mode)

        m = muladd((k)ᵀ → T, q → T, ct.zeros((TILE_I, TILE_L), Float32))
        n = muladd((u)ᵀ → T, q → T, ct.zeros((TILE_I, TILE_L), Float32))

        sig = 1 ./ (1 .+ exp.(0 .- m))
        silu_m = m .* sig
 
        ā = muladd((v)ᵀ → T, ō → T, ct.zeros((TILE_I, TILE_L), Float32))

        dsilu_dm = sig .* (1 .+ m .* (1 .- sig))

        M̄ = ā .* n .* dsilu_dm
        N̄ = ā .* silu_m

        q̄_acc = muladd(k → T, M̄ → T, q̄_acc)
        q̄_acc = muladd(u → T, N̄ → T, q̄_acc)

        i += 1i32
    end

    ct.store(Q̄, (1, h, j), reshape(q̄_acc, (TILE_D, 1, TILE_L)) → eltype(Q̄))

    return
end

function mhffn_bwd_dkuv(
    Q::TileArray3, K::TileArray3, U::TileArray3, V::TileArray3,
    Ō::TileArray3, K̄::TileArray3, Ū::TileArray3, V̄::TileArray3,
    T::Type,
    TILE_D::Int, TILE_L::Int, TILE_I::Int,
)
    padding_mode = ct.PaddingMode.Zero
    i, h = ct.bid(1), ct.bid(2)

    k = ct.load(K, (1, i, h), (TILE_D, TILE_I); padding_mode)
    u = ct.load(U, (1, i, h), (TILE_D, TILE_I); padding_mode)
    v = ct.load(V, (1, i, h), (TILE_D, TILE_I); padding_mode)

    k̄_acc = ct.zeros((TILE_D, TILE_I), Float32)
    ū_acc = ct.zeros((TILE_D, TILE_I), Float32)
    v̄_acc = ct.zeros((TILE_D, TILE_I), Float32)

    num_j = cld(size(Q, 3), TILE_L)
    j = 1i32
    while j <= num_j
        q = dropdims(ct.load(Q, (1, h, j), (TILE_D, 1, TILE_L); padding_mode), dims=2)
        ō = dropdims(ct.load(Ō, (1, h, j), (TILE_D, 1, TILE_L); padding_mode), dims=2)

        m = muladd((k)ᵀ → T, q → T, ct.zeros((TILE_I, TILE_L), Float32))
        n = muladd((u)ᵀ → T, q → T, ct.zeros((TILE_I, TILE_L), Float32))

        sig = 1 ./ (1 .+ exp.(0 .- m))
        silu_m = m .* sig

        a = silu_m .* n

        ā = muladd((v)ᵀ → T, ō → T, ct.zeros((TILE_I, TILE_L), Float32))

        dsilu_dm = sig .* (1 .+ m .* (1 .- sig))

        M̄ = ā .* n .* dsilu_dm
        N̄ = ā .* silu_m

        k̄_acc = muladd(q → T, (M̄)ᵀ → T, k̄_acc)
        ū_acc = muladd(q → T, (N̄)ᵀ → T, ū_acc)
        v̄_acc = muladd(ō → T, (a)ᵀ → T, v̄_acc)

        j += 1i32
    end

    ct.store(K̄, (1, i, h), k̄_acc → eltype(K̄))
    ct.store(Ū, (1, i, h), ū_acc → eltype(Ū))
    ct.store(V̄, (1, i, h), v̄_acc → eltype(V̄))

    return
end

function multihead_ffn!(O,
    Q, K, U, V;
    compute = eltype(Q),
    verify = nothing
)
    TILE_D, H, L = size(Q)

    key = (eltype(Q), compute, TILE_D)

    autotune_launch(mhf_fwd,
        CartesianSpace(TILE_L=(32, 64, 128), TILE_I=(32, 64, 128), occupancy=(1, 2, 4)),
        cfg -> (cld(L, cfg.TILE_L), H),
        cfg -> (
            Q, K, U, V, O,
            Constant(compute),
            Constant(TILE_D), Constant(TILE_L), Constant(TILE_I)
        );
        key, verify
    )

    return ()
end

function ∇multihead_ffn!(Q̄, K̄, Ū, V̄,
    Ō, Q, K, U, V;
    compute = eltype(Q),
    verify = nothing
)
    TILE_D, H, L = size(Q)

    key = (eltype(Q), compute, TILE_D)

    autotune_launch(mhffn_bwd_dq,
        CartesianSpace(TILE_L=(32, 64, 128), TILE_I=(32, 64, 128), occupancy=(1, 2, 4)),
        cfg -> (cld(L, cfg.TILE_L), H),
        cfg -> (
            Q, K, U, V, Ō, Q̄,
            Constant(compute),
            Constant(TILE_D), Constant(TILE_L), Constant(TILE_I)
        );
        key, verify
    )

    autotune_launch(mhffn_bwd_dkuv,
        CartesianSpace(TILE_L=(32, 64, 128), TILE_I=(32, 64, 128), occupancy=(1, 2, 4)),
        cfg -> (cld(de, cfg.TILE_I), H),
        cfg -> (
            Q, K, U, V, Ō, K̄, Ū, V̄,
            Constant(compute),
            Constant(TILE_D), Constant(TILE_L), Constant(TILE_I)
        );
        key, verify
    )

    return nothing
end


#=
function CRC.rrule(::typeof(mhf), Q, K, U, V)
    O = mhf_fwd(Q, K, U, V)
    function mhf_pullback(Ō)
        Q̄, K̄, Ū, V̄ = mhf_bwd(Q, K, U, V, CRC.unthunk(Ō))
        return CRC.NoTangent(), Q̄, K̄, Ū, V̄
    end
    return O, mhf_pullback
end
=#
