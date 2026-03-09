# scores: (H, L) — raw router logits
# I_out:  (k, L) — top-k head indices (Int32, 1-indexed)
# V_out:  (k, L) — raw selected scores (not normalized)

function top_k_gating_fwd(
    scores::TileMatrix,
    I_out::TileMatrix{Int32},
    V_out::TileMatrix,
    Tacc::Type,
    H::Int, k::Int, TILE_H::Int, TILE_L::Int,
)
    padding_mode = ct.PaddingMode.Zero
    bid = ct.bid(1)

    s = ct.load(scores, (1i32, bid), (TILE_H, TILE_L); padding_mode)
    h_2d = reshape(ct.arange((TILE_H,), Int32), (TILE_H, 1))

    # Mask padded rows so they never win argmax
    s = ifelse.(h_2d .> H, Tacc(-Inf32), s)

    # Top-k: iterative argmax + mask
    j = 1i32
    while j <= k
        val_j = maximum(s; dims=1)       # (1, TILE_L)
        idx_j = argmax(s; dims=1)        # (1, TILE_L) Int32

        ct.store(I_out, (j, bid), idx_j)
        ct.store(V_out, (j, bid), val_j → eltype(V_out))

        s = ifelse.(h_2d .== idx_j, Tacc(-Inf32), s)
        j += 1i32
    end

    return
end

# ── Backward kernel ──────────────────────────────────────────────────
# Scatter upstream gradients from (k, L) back to (H, L) at selected indices.

function top_k_gating_bwd(
    DV::TileMatrix,
    I::TileMatrix{Int32},
    dS::TileMatrix,
    Tacc::Type,
    TILE_H::Int, k::Int, TILE_L::Int,
)
    padding_mode = ct.PaddingMode.Zero
    bid = ct.bid(1)

    h_2d = reshape(ct.arange((TILE_H,), Int32), (TILE_H, 1))

    dscores = ct.zeros((TILE_H, TILE_L), Tacc)
    j = 1i32
    while j <= k
        dv_j  = ct.load(DV, (j, bid), (1, TILE_L); padding_mode)
        idx_j = ct.load(I, (j, bid), (1, TILE_L); padding_mode)

        mask = h_2d .== idx_j                  # (TILE_H, TILE_L)
        dscores = dscores .+ ifelse.(mask, dv_j, Tacc(0))

        j += 1i32
    end

    ct.store(dS, (1i32, bid), dscores → eltype(dS))
    return
end

# ── Launchers ────────────────────────────────────────────────────────

function top_k_gating_fwd!(I_out, V_out, scores; k,
    verify = nothing,
)
    H, L = size(scores)
    T = eltype(scores)
    Tacc = arithmetic_type(T)
    TILE_H = nextpow(2, H)

    key = (T, Tacc, H, k, nextpow(4, L))

    autotune_launch(top_k_gating_fwd,
        CartesianSpace(TILE_L=(32, 64, 128), occupancy=(1, 2, 4)),
        cfg -> (cld(L, cfg.TILE_L),),
        cfg -> (
            scores, I_out, V_out,
            Constant(Tacc),
            Constant(H), Constant(k), Constant(TILE_H), Constant(cfg.TILE_L),
        );
        key, verify,
    )

    return nothing
end

function top_k_gating_bwd!(dS, DV, I; k,
    verify = nothing,
)
    H, L = size(dS)
    T = eltype(DV)
    Tacc = arithmetic_type(T)
    TILE_H = nextpow(2, H)

    key = (:bwd, T, Tacc, H, k, nextpow(4, L))

    autotune_launch(top_k_gating_bwd,
        CartesianSpace(TILE_L=(32, 64, 128), occupancy=(1, 2, 4)),
        cfg -> (cld(L, cfg.TILE_L),),
        cfg -> (
            DV, I, dS,
            Constant(Tacc),
            Constant(TILE_H), Constant(k), Constant(cfg.TILE_L),
        );
        key, verify,
    )

    return nothing
end
