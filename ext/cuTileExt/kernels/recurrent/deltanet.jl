# Gated DeltaNet recurrent state update kernel (decode step)
#
# Translated from triton_deltanet_recurrent.py. Two-pass tiled design:
#   Pass 1: Decay state + accumulate S^T @ k
#   Pass 2: Rank-1 update (delta rule) + output query S^T @ q
#
# Grid: (H, B) — one block per (head, batch)
# State S is [Dk, Dv, H, B] mutated in-place

function deltanet_recurrent_fwd(
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
    h = ct.bid(1)
    b = ct.bid(2)

    # Load scalar gate and beta for this (head, batch)
    g = ct.load(Gate, (h, b), (1, 1)) → Float32
    decay = exp.(g)
    beta = ct.load(Beta, (h, b), (1, 1)) → Float32

    # Load value vector [Dv]
    v = ct.load(V, (1, h, b), (Dv, 1, 1)) → Float32
    v = reshape(v, (Dv,))

    num_dk_tiles = cld(Int32(Dk), Int32(BLOCK_DK))

    # ===== Pass 1: Decay state + accumulate S^T @ k =====
    accumulated = ct.zeros((Dv,), Float32)
    dk_tile = 1i32
    while dk_tile <= num_dk_tiles
        # Load state tile [BLOCK_DK, Dv]
        s_tile = ct.load(S, (dk_tile, 1, h, b), (BLOCK_DK, Dv); padding_mode=ct.PaddingMode.Zero) → Float32

        # Decay
        s_tile = s_tile .* reshape(decay, (1, 1))

        # Load k tile [BLOCK_DK]
        k_tile = ct.load(K, (dk_tile, h, b), (BLOCK_DK, 1, 1); padding_mode=ct.PaddingMode.Zero) → Float32
        k_tile = reshape(k_tile, (BLOCK_DK,))

        # Accumulate S^T @ k: sum over BLOCK_DK dimension
        accumulated = accumulated .+ reshape(sum(s_tile .* reshape(k_tile, (BLOCK_DK, 1)), dims=1), (Dv,))

        # Store decayed state
        ct.store(S, (dk_tile, 1, h, b), s_tile → eltype(S))

        dk_tile += 1i32
    end

    # Delta = beta * (v - S^T @ k)
    delta = reshape(beta, (1,)) .* (v .- accumulated)

    # ===== Pass 2: Rank-1 update + output query =====
    output = ct.zeros((Dv,), Float32)
    dk_tile = 1i32
    while dk_tile <= num_dk_tiles
        # Reload decayed state (should be in L2 cache)
        s_tile = ct.load(S, (dk_tile, 1, h, b), (BLOCK_DK, Dv); padding_mode=ct.PaddingMode.Zero) → Float32

        # Load k tile
        k_tile = ct.load(K, (dk_tile, h, b), (BLOCK_DK, 1, 1); padding_mode=ct.PaddingMode.Zero) → Float32
        k_tile = reshape(k_tile, (BLOCK_DK,))

        # Rank-1 update: S += outer(k, delta)
        s_tile = s_tile .+ reshape(k_tile, (BLOCK_DK, 1)) .* reshape(delta, (1, Dv))

        # Store updated state
        ct.store(S, (dk_tile, 1, h, b), s_tile → eltype(S))

        # Load q tile for output query
        q_tile = ct.load(Q, (dk_tile, h, b), (BLOCK_DK, 1, 1); padding_mode=ct.PaddingMode.Zero) → Float32
        q_tile = reshape(q_tile, (BLOCK_DK,))

        # Accumulate output: S^T @ q
        output = output .+ reshape(sum(s_tile .* reshape(q_tile, (BLOCK_DK, 1)), dims=1), (Dv,))

        dk_tile += 1i32
    end

    ct.store(O, (1, h, b), reshape(output → eltype(O), (Dv, 1, 1)))

    return
end

function deltanet_recurrent_step!(O, Q, K, V, Beta, Gate, S; verify=nothing)
    Dk, H, B = size(Q)
    Dv = size(V, 1)

    key = (eltype(Q), Dk, Dv)

    autotune_launch(deltanet_recurrent_fwd,
        CartesianSpace(BLOCK_DK=(8, 16, 32, 64, 128)),
        cfg -> (H, B),
        cfg -> (
            Q, K, V, Beta, Gate, S, O,
            Constant(Dk), Constant(Dv),
            Constant(cfg.BLOCK_DK),
        );
        key, verify
    )
end
