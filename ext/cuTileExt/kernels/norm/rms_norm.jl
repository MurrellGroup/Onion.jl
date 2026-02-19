#=============================================================================
 RMSNorm Forward Kernel

 Forward pass: computes RMS, normalizes input, and applies weighted scaling.
 Column-major: X is (M, N) with M=features, N=batch. Normalizes along dim 1.

   y = x * rstd * (w + offset)
   rstd = 1/sqrt(mean(x²) + eps)

 Simpler than LayerNorm: no mean subtraction, no bias.
 Zero-padded loads contribute 0² = 0 to sum of squares, so no masking needed.

 Args:
     X: Input tensor (M, N).
     W: Weight tensor (M,).
     Y: Output tensor (M, N).
     Rstd: Output reciprocal standard deviation tensor (N,).
     offset: Scalar offset added to weight.
     eps: Epsilon for numerical stability.
     TILE_M: Tile size along M (feature) dimension.
=============================================================================#
function rms_norm_fwd(
    X::TileMatrix{Float32}, W::TileVector{Float32},
    Y::TileMatrix{Float32}, Rstd::TileVector{Float32},
    offset::Float32, eps::Float32, TILE_M::Int
)
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)

    # Compute sum of squares
    ss = ct.full((1, TILE_M), 0.0f0, Float32)
    i = 1i32
    while i <= num_tiles
        tx = reshape(ct.load(X, (i, bid_n), (TILE_M, 1); padding_mode=ct.PaddingMode.Zero), (1, TILE_M))
        ss = ss .+ (tx .* tx)
        i += 1i32
    end
    rstd = 1.0f0 ./ sqrt.(sum(ss; dims=2) / M .+ eps)
    ct.store(Rstd, bid_n, rstd)

    # Normalize and apply weight
    i = 1i32
    while i <= num_tiles
        tx = reshape(ct.load(X, (i, bid_n), (TILE_M, 1); padding_mode=ct.PaddingMode.Zero), (1, TILE_M))
        tw = reshape(ct.load(W, i, (TILE_M,); padding_mode=ct.PaddingMode.Zero), (1, TILE_M))
        ty = tx .* rstd .* (tw .+ offset)
        ct.store(Y, (i, bid_n), reshape(ty, (TILE_M, 1)))
        i += 1i32
    end

    return
end

#=============================================================================
 RMSNorm Backward Kernels

 Backward pass: computes gradients for RMSNorm.
 Column-major: X is (M, N) with M=features, N=batch.

   dd = Σ_m(dy_m * (w_m + offset) * x_m)
   dx = rstd * (dy * (w + offset) - rstd² * dd/M * x)
   dw = Σ_n(dy * x * rstd)   (summed over batch)

 Two kernels:
 1. rms_norm_bwd_dx_partial_dw - Computes dX and partial dW
 2. rms_norm_bwd_dw - Final reduction for dW
=============================================================================#

"""
    rms_norm_bwd_dx_partial_dw(DX, DY, DW, X, W, Rstd, Locks, offset, N_GROUPS, TILE_M)

Backward pass part 1: computes dX and partial dW.

Args:
    DX: Output gradient with respect to X (M, N).
    DY: Input gradient with respect to Y (M, N).
    DW: Partial gradient with respect to W (M, N_GROUPS).
    X: Input tensor (M, N).
    W: Weight tensor (M,).
    Rstd: Reciprocal standard deviation tensor (N,).
    Locks: Lock tensor for atomic operations (N_GROUPS,).
    offset: Scalar offset added to weight.
    N_GROUPS: Number of partial gradient groups.
    TILE_M: Tile size along M (feature) dimension.
"""
function rms_norm_bwd_dx_partial_dw(
    DX::TileMatrix{Float32}, DY::TileMatrix{Float32},
    DW::TileMatrix{Float32},
    X::TileMatrix{Float32}, W::TileVector{Float32},
    Rstd::TileVector{Float32},
    Locks::TileVector{Int},
    offset::Float32, N_GROUPS::Int, TILE_M::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)
    group_id = ((bid_n - 1i32) % Int32(N_GROUPS)) + 1i32

    rstd = ct.load(Rstd, bid_n, (1,); padding_mode)

    # First pass: compute dd = sum(dy * (w + offset) * x) / M
    dd = ct.full((1, TILE_M), 0.0f0, Float32)
    i = 1i32
    while i <= num_tiles
        tx = reshape(ct.load(X, (i, bid_n), (TILE_M, 1); padding_mode), (1, TILE_M))
        tw = reshape(ct.load(W, i, (TILE_M,); padding_mode), (1, TILE_M))
        tdy = reshape(ct.load(DY, (i, bid_n), (TILE_M, 1); padding_mode), (1, TILE_M))
        dd = dd .+ (tdy .* (tw .+ offset) .* tx)
        i += 1i32
    end
    dd = sum(dd; dims=2) / M

    # Second pass: compute dx and partial dw
    i = 1i32
    while i <= num_tiles
        tx = reshape(ct.load(X, (i, bid_n), (TILE_M, 1); padding_mode), (1, TILE_M))
        tw = reshape(ct.load(W, i, (TILE_M,); padding_mode), (1, TILE_M))
        tdy = reshape(ct.load(DY, (i, bid_n), (TILE_M, 1); padding_mode), (1, TILE_M))

        # dx = rstd * (dy * (w + offset) - rstd² * dd * x)
        tdx = rstd .* (tdy .* (tw .+ offset) .- (rstd .* rstd) .* dd .* tx)
        ct.store(DX, (i, bid_n), reshape(tdx, (TILE_M, 1)))

        # dw = dy * x * rstd (partial, accumulated across batch)
        partial_dw = reshape(tdy .* tx .* rstd, (TILE_M, 1))

        # Acquire spinlock
        while ct.atomic_cas(Locks, group_id, 0, 1;
                           memory_order=ct.MemoryOrder.Acquire) == 1
            # spin
        end

        # Critical section: accumulate partial gradients
        partial_dw = partial_dw .+ ct.load(DW, (i, group_id), (TILE_M, 1); padding_mode)
        ct.store(DW, (i, group_id), partial_dw)

        # Release spinlock
        ct.atomic_xchg(Locks, group_id, 0;
                      memory_order=ct.MemoryOrder.Release)

        i += 1i32
    end

    return
end

"""
    rms_norm_bwd_dw(DW, FINAL_DW, TILE_G, TILE_F)

Backward pass part 2: Final reduction for dW.

Args:
    DW: Partial gradient with respect to W (M, N_GROUPS).
    FINAL_DW: Final gradient with respect to W (M,).
    TILE_G: Tile size along groups dimension (dim 2).
    TILE_F: Tile size along feature dimension (dim 1).
"""
function rms_norm_bwd_dw(
    DW::TileMatrix{Float32},
    FINAL_DW::TileVector{Float32},
    TILE_G::Int, TILE_F::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid = ct.bid(1)
    num_tiles = ct.num_tiles(DW, 2, (TILE_F, TILE_G))

    dw = ct.zeros((TILE_F, TILE_G), Float32)
    i = 1i32
    while i <= num_tiles
        dw = dw .+ ct.load(DW, (bid, i), (TILE_F, TILE_G); padding_mode)
        i += 1i32
    end
    sum_dw = sum(dw; dims=2)

    ct.store(FINAL_DW, bid, sum_dw)

    return
end

function rms_norm(
    X::AbstractMatrix{T}, W::AbstractVector{Tw};
    eps,
    offset = 0f0,
    verify = nothing
) where {T, Tw}
    M, N = size(X)
    @assert length(W) == M

    Y = similar(X)
    Rstd = similar(X, Float32, N)

    key = (T, Tw)

    autotune_launch(rms_norm_fwd,
        CartesianSpace(TILE_M=(128, 256, 512, 1024)),
        cfg -> N,
        cfg -> (
            X, W, Y, Rstd,
            Constant(Float32(offset)), Constant(Float32(eps)), Constant(cfg.TILE_M)
        );
        key, verify
    )

    return Y, Rstd
end

function ∇rms_norm(
    Ȳ::AbstractMatrix, X::AbstractMatrix,
    W::AbstractVector, Rstd::AbstractVector;
    offset = 0f0,
    N_GROUPS = 64, # XXX: autotune this
)
    M, N = size(X)

    X̄ = similar(X)
    W̄_partial = fill!(similar(W, M, N_GROUPS), 0)
    Locks = fill!(similar(X, Int, N_GROUPS), 0)
    W̄ = similar(W)

    key = (eltype(X), eltype(W))

    autotune_launch(rms_norm_bwd_dx_partial_dw,
        CartesianSpace(TILE_M=(128, 256, 512, 1024)),
        cfg -> N,
        cfg -> (
            X̄, Ȳ, fill!(W̄_partial, 0), X, W, Rstd, fill!(Locks, 0),
            Constant(Float32(offset)), Constant(N_GROUPS), Constant(cfg.TILE_M)
        )
    )

    autotune_launch(rms_norm_bwd_dw,
        CartesianSpace(TILE_F=(128, 256, 512, 1024), TILE_G=(32,)),
        cfg -> cld(M, cfg.TILE_F),
        cfg -> (
            W̄_partial, W̄,
            Constant(cfg.TILE_G), Constant(cfg.TILE_F)
        );
        key,
    )

    return X̄, W̄
end
