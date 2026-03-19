function deltanet_recurrent_decode_fwd(
    Q::TileArray3,     # (Dk, H, B)
    K::TileArray3,     # (Dk, H, B)
    V::TileArray3,     # (Dv, H, B)
    Beta::TileMatrix,  # (H, B)
    Gate::TileMatrix,  # (H, B)
    S::TileArray4,     # (Dk, Dv, H, B) — state, mutated
    O::TileArray3,     # (Dv, H, B) — output
    Dk::Int, Dv::Int,
    BLOCK_DK::Int,
)
    padding_mode = ct.PaddingMode.Zero
    h, b = ct.bid(1), ct.bid(2)

    g = Gate[h, b] → Float32
    decay = exp(g)
    beta = Beta[h, b] → Float32

    v = ct.load(V, (1, h, b), (Dv,)) → Float32

    num_tiles = cld(Int32(Dk), Int32(BLOCK_DK))

    # Decay state + accumulate S^T @ k
    acc = ct.zeros((Dv,), Float32)
    for i in 1i32:num_tiles
        s = ct.load(S, (i, 1, h, b), (BLOCK_DK, Dv); padding_mode) → Float32
        k = ct.load(K, (i, h, b), (BLOCK_DK,); padding_mode) → Float32

        s = s .* decay
        ct.store(S, (i, 1, h, b), s → eltype(S))

        acc = mva((s)ᵀ, k, acc)
    end

    delta = beta .* (v .- acc)

    # Rank-1 update + output query
    output = ct.zeros((Dv,), Float32)
    for i in 1i32:num_tiles
        s = ct.load(S, (i, 1, h, b), (BLOCK_DK, Dv); padding_mode) → Float32
        k = ct.load(K, (i, h, b), (BLOCK_DK,); padding_mode) → Float32

        s = s .+ k .* (delta)ᵀ
        ct.store(S, (i, 1, h, b), s → eltype(S))

        q = ct.load(Q, (i, h, b), (BLOCK_DK,); padding_mode) → Float32
        output = mva((s)ᵀ, q, output)
    end

    ct.store(O, (1, h, b), output → eltype(O))

    return
end

function deltanet_recurrent_decode_step!(O, Q, K, V, Beta, Gate, S; verify=nothing)
    Dk, H, B = size(Q)
    Dv = size(V, 1)

    key = (eltype(Q), Dk, Dv)

    autotune_launch(deltanet_recurrent_decode_fwd,
        CartesianSpace(BLOCK_DK=(8, 16, 32)),
        cfg -> (H, B),
        cfg -> (
            Q, K, V, Beta, Gate, S, O,
            Constant(Dk), Constant(Dv),
            Constant(cfg.BLOCK_DK),
        );
        key, verify
    )
end
