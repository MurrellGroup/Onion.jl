# Fused sequential DeltaNet recurrence kernel.
# Grid: (H, B) — one block per head per batch, loops over T.
# Same algorithm as decode kernel but fused over the sequence length.

function deltanet_sequence_fwd(
    Q::TileArray4,      # (Dk, T, H, B)
    K::TileArray4,      # (Dk, T, H, B)
    V::TileArray4,      # (Dv, T, H, B)
    Beta::TileArray3,   # (H, T, B)
    Gate::TileArray3,   # (H, T, B)
    S::TileArray4,      # (Dk, Dv, H, B) — state, mutated
    O::TileArray4,      # (Dv, T, H, B) — output
    Dk::Int, Dv::Int, T_len::Int,
    BLOCK_DK::Int,
)
    padding_mode = ct.PaddingMode.Zero
    h, b = ct.bid(1), ct.bid(2)
    num_dk_tiles = cld(Int32(Dk), Int32(BLOCK_DK))

    for t in 1i32:T_len
        g = Gate[h, t, b] → Float32
        decay = exp(g)
        beta = Beta[h, t, b] → Float32

        v = ct.load(V, (1, t, h, b), (Dv,)) → Float32

        # Pass 1: Decay state + accumulate S^T @ k
        acc = zeros(Float32, Dv)
        for i in 1i32:num_dk_tiles
            s = ct.load(S, (i, 1, h, b), (BLOCK_DK, Dv); padding_mode) → Float32
            k = ct.load(K, (i, t, h, b), (BLOCK_DK,); padding_mode) → Float32

            s = s .* decay
            ct.store(S, (i, 1, h, b), s → eltype(S))

            acc = mva((s)ᵀ, k, acc)
        end

        delta = beta .* (v .- acc)

        # Pass 2: Rank-1 update + output query
        output = zeros(Float32, Dv)
        for i in 1i32:num_dk_tiles
            s = ct.load(S, (i, 1, h, b), (BLOCK_DK, Dv); padding_mode) → Float32
            k = ct.load(K, (i, t, h, b), (BLOCK_DK,); padding_mode) → Float32

            s = s .+ k .* (delta)ᵀ
            ct.store(S, (i, 1, h, b), s → eltype(S))

            q = ct.load(Q, (i, t, h, b), (BLOCK_DK,); padding_mode) → Float32
            output = mva((s)ᵀ, q, output)
        end

        ct.store(O, (1, t, h, b), output → eltype(O))
    end

    return
end

function deltanet_sequence_step!(O, Q, K, V, Beta, Gate, S; verify=nothing)
    Dk, T, H, B = size(Q)
    Dv = size(V, 1)

    key = (eltype(Q), Dk, Dv, T)

    autotune_launch(deltanet_sequence_fwd,
        CartesianSpace(BLOCK_DK=(8, 16, 32)),
        cfg -> (H, B),
        cfg -> (
            Q, K, V, Beta, Gate, S, O,
            Constant(Dk), Constant(Dv), Constant(T),
            Constant(cfg.BLOCK_DK),
        );
        key, verify
    )
end
