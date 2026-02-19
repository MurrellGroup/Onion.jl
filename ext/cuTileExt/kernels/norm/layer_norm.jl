#=============================================================================
 LayerNorm Forward Kernel

 Forward pass: computes mean/var, normalizes input, and applies affine transform.

 Args:
     X: Input tensor (M, N).
     W: Weight tensor (N,).
     B: Bias tensor (N,).
     Y: Output tensor (M, N).
     Mean: Output mean tensor (M,).
     Rstd: Output reciprocal standard deviation tensor (M,).
     eps: Epsilon for numerical stability.
     TILE_N: Tile size along N dimension.
=============================================================================#
function layer_norm_fwd(
    X::TileMatrix{Float32}, W::TileVector{Float32},
    B::TileVector{Float32}, Y::TileMatrix{Float32},
    Mean::TileVector{Float32}, Rstd::TileVector{Float32},
    eps::Float32, TILE_N::Int
)
    bid_m = ct.bid(1)
    num_tiles = ct.num_tiles(X, 2, (1, TILE_N))
    N = size(X, 2)

    # Compute mean
    mean = ct.full((1, TILE_N), 0.0f0, Float32)
    j = Int32(1)
    while j <= num_tiles
        tx = ct.load(X, (bid_m, j), (1, TILE_N); padding_mode=ct.PaddingMode.Zero)
        mean = mean .+ tx
        j += Int32(1)
    end
    mean = sum(mean; dims=2) / N
    ct.store(Mean, bid_m, mean)

    # Compute variance
    var = ct.full((1, TILE_N), 0.0f0, Float32)
    j = Int32(1)
    while j <= num_tiles
        tx = ct.load(X, (bid_m, j), (1, TILE_N); padding_mode=ct.PaddingMode.Zero)
        # Mask for valid elements
        mask = ct.broadcast_to(((j - Int32(1)) * Int32(TILE_N) .+ ct.arange((TILE_N,), Int32)) .<= N, (1, TILE_N))
        centered_tx = ifelse.(mask, tx .- mean, 0.0f0)
        var = var .+ (centered_tx .^ 2.0f0)
        j += Int32(1)
    end
    var = sum(var; dims=2) / N
    rstd = 1.0f0 ./ sqrt.(var .+ eps)
    ct.store(Rstd, bid_m, rstd)

    # Normalize and apply affine transformation
    j = Int32(1)
    while j <= num_tiles
        tx = ct.load(X, (bid_m, j), (1, TILE_N); padding_mode=ct.PaddingMode.Zero)
        tw = reshape(ct.load(W, j, (TILE_N,); padding_mode=ct.PaddingMode.Zero), (1, TILE_N))
        tb = reshape(ct.load(B, j, (TILE_N,); padding_mode=ct.PaddingMode.Zero), (1, TILE_N))
        ty = (tx .- mean) .* rstd
        ty = ty .* tw .+ tb
        ct.store(Y, (bid_m, j), ty)
        j += Int32(1)
    end

    return
end

#=============================================================================
 LayerNorm Backward Kernels

 Backward pass: computes gradients for LayerNorm.
 The full backward pass has two kernels:
 1. layer_norm_bwd_dx - Computes dX (gradient with respect to input)
 2. layer_norm_bwd_dwdb - Computes dW and dB (requires atomic accumulation)

 For now, we implement a simplified backward that just computes dX.
=============================================================================#

"""
Helper function for backward pass - loads data and computes common terms.
This gets inlined by Julia's compiler.
bid_m and j are 1-indexed (block ID and tile index).
"""
@inline function bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N)
    padding_mode = ct.PaddingMode.Zero
    tx = ct.load(X, (bid_m, j), (1, TILE_N); padding_mode)
    tw = reshape(ct.load(W, j, (TILE_N,); padding_mode), (1, TILE_N))
    tdy = ct.load(DY, (bid_m, j), (1, TILE_N); padding_mode)
    xhat = (tx .- mean) .* rstd
    wdy = tw .* tdy

    # Mask for valid elements
    indices = ct.arange((TILE_N,), Int32)
    offset = (j - Int32(1)) * Int32(TILE_N)
    global_indices = offset .+ indices
    mask = ct.broadcast_to(global_indices .<= N, (1, TILE_N))

    xhat_masked = ifelse.(mask, xhat, 0.0f0)
    wdy_masked = ifelse.(mask, wdy, 0.0f0)

    return tdy, xhat_masked, wdy_masked
end

"""
    layer_norm_bwd_dx(DX, DY, X, W, Mean, Rstd, TILE_N)

Backward pass: computes gradient with respect to input X.

Args:
    DX: Output gradient with respect to X (M, N).
    DY: Input gradient with respect to Y (M, N).
    X: Input tensor (M, N).
    W: Weight tensor (N,).
    Mean: Mean tensor (M,).
    Rstd: Reciprocal standard deviation tensor (M,).
    TILE_N: Tile size along N dimension.
"""
function layer_norm_bwd_dx(
    DX::TileMatrix{Float32}, DY::TileMatrix{Float32},
    X::TileMatrix{Float32}, W::TileVector{Float32},
    Mean::TileVector{Float32}, Rstd::TileVector{Float32},
    TILE_N::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid_m = ct.bid(1)
    num_tiles = ct.num_tiles(X, 2, (1, TILE_N))
    N = size(X, 2)

    # Load mean and rstd for this row
    rstd = ct.load(Rstd, bid_m, (1,); padding_mode)
    mean = ct.load(Mean, bid_m, (1,); padding_mode)

    # First pass: compute c1 and c2 reduction terms
    c1 = ct.full((1, TILE_N), 0.0f0, Float32)
    c2 = ct.full((1, TILE_N), 0.0f0, Float32)
    j = Int32(1)
    while j <= num_tiles
        _, xhat, wdy = bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N)
        c1 = c1 .+ (xhat .* wdy)
        c2 = c2 .+ wdy
        j += Int32(1)
    end
    c1 = sum(c1; dims=2) / N
    c2 = sum(c2; dims=2) / N

    # Second pass: compute dX
    j = Int32(1)
    while j <= num_tiles
        _, xhat, wdy = bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N)
        tdx = (wdy .- (xhat .* c1 .+ c2)) .* rstd
        ct.store(DX, (bid_m, j), tdx)
        j += Int32(1)
    end

    return
end

"""
    layer_norm_bwd_dx_partial_dwdb(DX, DY, DW, DB, X, W, Mean, Rstd, Locks, GROUP_SIZE_M, TILE_N)

Backward pass part 1: computes dX and partial dW/dB.
Accumulates partial gradients using atomic locks.

Args:
    DX: Output gradient with respect to X (M, N).
    DY: Input gradient with respect to Y (M, N).
    DW: Partial gradient with respect to W (N, GROUP_SIZE_M).
    DB: Partial gradient with respect to B (N, GROUP_SIZE_M).
    X: Input tensor (M, N).
    W: Weight tensor (N,).
    Mean: Mean tensor (M,).
    Rstd: Reciprocal standard deviation tensor (M,).
    Locks: Lock tensor for atomic operations (GROUP_SIZE_M,).
    GROUP_SIZE_M: Number of partial gradient groups.
    TILE_N: Tile size along N dimension.
"""
function layer_norm_bwd_dx_partial_dwdb(
    DX::TileMatrix{Float32}, DY::TileMatrix{Float32},
    DW::TileMatrix{Float32}, DB::TileMatrix{Float32},
    X::TileMatrix{Float32}, W::TileVector{Float32},
    Mean::TileVector{Float32}, Rstd::TileVector{Float32},
    Locks::TileVector{Int},
    GROUP_SIZE_M::Int, TILE_N::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid_m = ct.bid(1)
    num_tiles = ct.num_tiles(X, 2, (1, TILE_N))
    N = size(X, 2)
    group_bid_m = ((bid_m - Int32(1)) % Int32(GROUP_SIZE_M)) + Int32(1)

    # Load mean and rstd for this row
    mean = ct.load(Mean, bid_m, (1,); padding_mode)
    rstd = ct.load(Rstd, bid_)

    # First pass: compute c1 and c2 reduction terms
    c1 = ct.full((1, TILE_N), 0.0f0, Float32)
    c2 = ct.full((1, TILE_N), 0.0f0, Float32)
    j = Int32(1)
    while j <= num_tiles
        _, xhat, wdy = bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N)
        c1 = c1 .+ (xhat .* wdy)
        c2 = c2 .+ wdy
        j += Int32(1)
    end
    c1 = sum(c1; dims=2) / N
    c2 = sum(c2; dims=2) / N

    # Second pass: compute dX and partial dW/dB
    j = Int32(1)
    while j <= num_tiles
        tdy, xhat, wdy = bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N)
        tdx = (wdy .- (xhat .* c1 .+ c2)) .* rstd
        ct.store(DX, (bid_m, j), tdx)

        partial_dw = reshape(tdy .* xhat, (TILE_N, 1))
        partial_db = reshape(tdy, (TILE_N, 1))

        # Acquire spinlock
        while ct.atomic_cas(Locks, group_bid_m, 0, 1;
                           memory_order=ct.MemoryOrder.Acquire) == 1
            # spin
        end

        # Critical section: accumulate partial gradients
        partial_dw = partial_dw .+ ct.load(DW, (j, group_bid_m), (TILE_N, 1); padding_mode)
        partial_db = partial_db .+ ct.load(DB, (j, group_bid_m), (TILE_N, 1); padding_mode)
        ct.store(DW, (j, group_bid_m), partial_dw)
        ct.store(DB, (j, group_bid_m), partial_db)

        # Release spinlock
        ct.atomic_xchg(Locks, group_bid_m, 0;
                      memory_order=ct.MemoryOrder.Release)

        j += Int32(1)
    end

    return
end

"""
    layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, TILE_M, TILE_N)

Backward pass part 2: Final reduction for dW and dB.

Args:
    DW: Partial gradient with respect to W (N, TILE_M).
    DB: Partial gradient with respect to B (N, TILE_M).
    FINAL_DW: Final gradient with respect to W (N,).
    FINAL_DB: Final gradient with respect to B (N,).
    TILE_M: Number of partial gradients to reduce.
    TILE_N: Tile size along N dimension.
"""
function layer_norm_bwd_dwdb(
    DW::TileMatrix{Float32}, DB::TileMatrix{Float32},
    FINAL_DW::TileVector{Float32}, FINAL_DB::TileVector{Float32},
    TILE_M::Int, TILE_N::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(DW, 2, (TILE_N, TILE_M))

    dw = ct.zeros((TILE_N, TILE_M), Float32)
    db = ct.zeros((TILE_N, TILE_M), Float32)
    i = Int32(1)
    while i <= num_tiles
        dw = dw .+ ct.load(DW, (bid_n, i), (TILE_N, TILE_M); padding_mode)
        db = db .+ ct.load(DB, (bid_n, i), (TILE_N, TILE_M); padding_mode)
        i += Int32(1)
    end
    sum_dw = sum(dw; dims=2)
    sum_db = sum(db; dims=2)

    ct.store(FINAL_DW, bid_n, sum_dw)
    ct.store(FINAL_DB, bid_n, sum_db)

    return
end

function layer_norm(
    X::AbstractMatrix{T}, W::AbstractVector{Tw}, B::AbstractVector{Tw};
    eps,
    verify = nothing
) where {T, Tw}
    M, N = size(x)
    @assert length(w) == length(b) == M

    # XXX: should X and Y be (N, M)? compare to python
    Y = similar(X)
    Mean = similar(X, M)
    Rstd = similar(X, M)

    key = (T, Tw)

    # TODO: GROUP_SIZE_M
    ct.autotune_launch(layer_norm_fwd,
        CartesianSpace(TILE_N=(128, 256, 512, 1024)),
        cfg -> M,
        cfg -> (
            X, W, B, Y, Mean, Rstd,
            Constant(eps), Constant(cfg.TILE_N)
        );
        key, verify
    )

    return Y, Mean, Rstd
end

function ∇layer_norm(
    Ȳ::AbstractMatrix, X::AbstractMatrix,
    W::AbstractVector, B::AbstractVector,
    Mean::AbstractVector, Rstd::AbstractVector;
    GROUP_SIZE_M = 64, # XXX: autotune this. difficult because two kernel launches
    # XXX: verify
)
    M, N = size(x)

    X̄ = similar(X)
    W̄_partial = fill!(similar(W, N, GROUP_SIZE_M), 0)
    B̄_partial = fill!(similar(B, N, GROUP_SIZE_M), 0)
    Locks = fill!(similar(X, Int, GROUP_SIZE_M), 0)
    W̄ = similar(W)
    B̄ = similar(B)

    ct.autotune_launch(layer_norm_bwd_dx_partial_dwdb,
        CartesianSpace(TILE_N=(128, 256, 512, 1024)),
        cfg -> M,
        cfg -> (
            X̄, Ȳ, W̄_partial, B̄_partial, X, W,
            Mean, Rstd, Locks,
            Constant(GROUP_SIZE_M),
            Constant(cfg.TILE_N)
        )
    )

    # TODO (cuTile.jl): result function to take args constructed from cfg?
    # (like {W̄,B̄}_partial with tuned GROUP_SIZE_M)
    ct.autotune_launch(layer_norm_bwd_dwdb,
        CartesianSpace(TILE_N=(128, 256, 512, 1024,), TILE_M=(32,)),
        cfg -> cld(N, cfg.TILE_N),
        cfg -> (
            W̄_partial, B̄_partial, W̄, B̄,
            Constant(TILE_M), Constant(TILE_N)
        );
        key,
    )

    return X̄
end
