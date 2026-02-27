#=============================================================================
 LayerNorm Forward Kernel

 Forward pass: computes mean/var, normalizes input, and applies affine transform.
 Column-major: X is (M, N) with M=features, N=batch. Normalizes along dim 1.

 Loads/stores use (TILE_M, 1) tile shape for column-major access.
 Internal computation uses (1, TILE_M) tiles (reshape is free in cuTile).

 Args:
     X: Input tensor (M, N).
     W: Weight tensor (M,).
     B: Bias tensor (M,).
     Y: Output tensor (M, N).
     Mean: Output mean tensor (N,).
     Rstd: Output reciprocal standard deviation tensor (N,).
     eps: Epsilon for numerical stability.
     TILE_M: Tile size along M (feature) dimension.
=============================================================================#
function layer_norm_fwd(
    X::TileMatrix{Float32}, W::TileVector{Float32},
    B::TileVector{Float32}, Y::TileMatrix{Float32},
    Mean::TileVector{Float32}, Rstd::TileVector{Float32},
    eps::Float32, TILE_M::Int
)
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)

    # Compute mean
    mean = ct.full((1, TILE_M), 0.0f0, Float32)
    i = 1i32
    while i <= num_tiles
        tx = reshape(ct.load(X, (i, bid_n), (TILE_M, 1); padding_mode=ct.PaddingMode.Zero), (1, TILE_M))
        mean = mean .+ tx
        i += 1i32
    end
    mean = sum(mean; dims=2) / M
    ct.store(Mean, bid_n, mean)

    # Compute variance
    var = ct.full((1, TILE_M), 0.0f0, Float32)
    i = 1i32
    while i <= num_tiles
        tx = reshape(ct.load(X, (i, bid_n), (TILE_M, 1); padding_mode=ct.PaddingMode.Zero), (1, TILE_M))
        # Mask for valid elements
        mask = reshape(((i - 1i32) * Int32(TILE_M) .+ ct.arange((TILE_M,), Int32)) .<= M, (1, TILE_M))
        centered_tx = ifelse.(mask, tx .- mean, 0.0f0)
        var = var .+ (centered_tx .^ 2.0f0)
        i += 1i32
    end
    var = sum(var; dims=2) / M
    rstd = 1.0f0 ./ sqrt.(var .+ eps)
    ct.store(Rstd, bid_n, rstd)

    # Normalize and apply affine transformation
    i = 1i32
    while i <= num_tiles
        tx = reshape(ct.load(X, (i, bid_n), (TILE_M, 1); padding_mode=ct.PaddingMode.Zero), (1, TILE_M))
        tw = reshape(ct.load(W, i, (TILE_M,); padding_mode=ct.PaddingMode.Zero), (1, TILE_M))
        tb = reshape(ct.load(B, i, (TILE_M,); padding_mode=ct.PaddingMode.Zero), (1, TILE_M))
        ty = (tx .- mean) .* rstd
        ty = ty .* tw .+ tb
        ct.store(Y, (i, bid_n), reshape(ty, (TILE_M, 1)))
        i += 1i32
    end

    return
end

#=============================================================================
 LayerNorm Backward Kernels

 Backward pass: computes gradients for LayerNorm.
 Column-major: X is (M, N) with M=features, N=batch.
 Same reshape convention as forward: (TILE_M, 1) at boundaries, (1, TILE_M) inside.

 The full backward pass has two kernels:
 1. layer_norm_bwd_dx_partial_dwdb - Computes dX and partial dW/dB
 2. layer_norm_bwd_dwdb - Final reduction for dW and dB
=============================================================================#

"""
Helper function for backward pass - loads data and computes common terms.
This gets inlined by Julia's compiler.
`bid_n` is the column (batch) index, `i` is the tile index along dim 1.
All returned tiles are in (1, TILE_M) computation orientation.
"""
@inline function bwd_helper(X, W, DY, bid_n, i, mean, rstd, TILE_M, M)
    padding_mode = ct.PaddingMode.Zero
    tx = reshape(ct.load(X, (i, bid_n), (TILE_M, 1); padding_mode), (1, TILE_M))
    tw = reshape(ct.load(W, i, (TILE_M,); padding_mode), (1, TILE_M))
    tdy = reshape(ct.load(DY, (i, bid_n), (TILE_M, 1); padding_mode), (1, TILE_M))
    xhat = (tx .- mean) .* rstd
    wdy = tw .* tdy

    # Mask for valid elements
    indices = ct.arange((TILE_M,), Int32)
    offset = (i - 1i32) * Int32(TILE_M)
    global_indices = offset .+ indices
    mask = reshape(global_indices .<= M, (1, TILE_M))

    xhat_masked = ifelse.(mask, xhat, 0.0f0)
    wdy_masked = ifelse.(mask, wdy, 0.0f0)

    return tdy, xhat_masked, wdy_masked
end

"""
    layer_norm_bwd_dx(DX, DY, X, W, Mean, Rstd, TILE_M)

Backward pass: computes gradient with respect to input X only.

Args:
    DX: Output gradient with respect to X (M, N).
    DY: Input gradient with respect to Y (M, N).
    X: Input tensor (M, N).
    W: Weight tensor (M,).
    Mean: Mean tensor (N,).
    Rstd: Reciprocal standard deviation tensor (N,).
    TILE_M: Tile size along M (feature) dimension.
"""
function layer_norm_bwd_dx(
    DX::TileMatrix{Float32}, DY::TileMatrix{Float32},
    X::TileMatrix{Float32}, W::TileVector{Float32},
    Mean::TileVector{Float32}, Rstd::TileVector{Float32},
    TILE_M::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)

    # Load mean and rstd for this column
    rstd = ct.load(Rstd, bid_n, (1,); padding_mode)
    mean = ct.load(Mean, bid_n, (1,); padding_mode)

    # First pass: compute c1 and c2 reduction terms
    c1 = ct.full((1, TILE_M), 0.0f0, Float32)
    c2 = ct.full((1, TILE_M), 0.0f0, Float32)
    i = 1i32
    while i <= num_tiles
        _, xhat, wdy = bwd_helper(X, W, DY, bid_n, i, mean, rstd, TILE_M, M)
        c1 = c1 .+ (xhat .* wdy)
        c2 = c2 .+ wdy
        i += 1i32
    end
    c1 = sum(c1; dims=2) / M
    c2 = sum(c2; dims=2) / M

    # Second pass: compute dX
    i = 1i32
    while i <= num_tiles
        _, xhat, wdy = bwd_helper(X, W, DY, bid_n, i, mean, rstd, TILE_M, M)
        tdx = (wdy .- (xhat .* c1 .+ c2)) .* rstd
        ct.store(DX, (i, bid_n), reshape(tdx, (TILE_M, 1)))
        i += 1i32
    end

    return
end

"""
    layer_norm_bwd_dx_partial_dwdb(DX, DY, DW, DB, X, W, Mean, Rstd, Locks, N_GROUPS, TILE_M)

Backward pass part 1: computes dX and partial dW/dB.
Accumulates partial gradients using atomic locks.

Args:
    DX: Output gradient with respect to X (M, N).
    DY: Input gradient with respect to Y (M, N).
    DW: Partial gradient with respect to W (M, N_GROUPS).
    DB: Partial gradient with respect to B (M, N_GROUPS).
    X: Input tensor (M, N).
    W: Weight tensor (M,).
    Mean: Mean tensor (N,).
    Rstd: Reciprocal standard deviation tensor (N,).
    Locks: Lock tensor for atomic operations (N_GROUPS,).
    N_GROUPS: Number of partial gradient groups.
    TILE_M: Tile size along M (feature) dimension.
"""
function layer_norm_bwd_dx_partial_dwdb(
    DX::TileMatrix{Float32}, DY::TileMatrix{Float32},
    DW::TileMatrix{Float32}, DB::TileMatrix{Float32},
    X::TileMatrix{Float32}, W::TileVector{Float32},
    Mean::TileVector{Float32}, Rstd::TileVector{Float32},
    Locks::TileVector{Int},
    N_GROUPS::Int, TILE_M::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)
    group_id = ((bid_n - 1i32) % Int32(N_GROUPS)) + 1i32

    # Load mean and rstd for this column
    mean = ct.load(Mean, bid_n, (1,); padding_mode)
    rstd = ct.load(Rstd, bid_n, (1,); padding_mode)

    # First pass: compute c1 and c2 reduction terms
    c1 = ct.full((1, TILE_M), 0.0f0, Float32)
    c2 = ct.full((1, TILE_M), 0.0f0, Float32)
    i = 1i32
    while i <= num_tiles
        _, xhat, wdy = bwd_helper(X, W, DY, bid_n, i, mean, rstd, TILE_M, M)
        c1 = c1 .+ (xhat .* wdy)
        c2 = c2 .+ wdy
        i += 1i32
    end
    c1 = sum(c1; dims=2) / M
    c2 = sum(c2; dims=2) / M

    # Second pass: compute dX and partial dW/dB
    i = 1i32
    while i <= num_tiles
        tdy, xhat, wdy = bwd_helper(X, W, DY, bid_n, i, mean, rstd, TILE_M, M)
        tdx = (wdy .- (xhat .* c1 .+ c2)) .* rstd
        ct.store(DX, (i, bid_n), reshape(tdx, (TILE_M, 1)))

        partial_dw = reshape(tdy .* xhat, (TILE_M, 1))
        partial_db = reshape(tdy, (TILE_M, 1))

        # Acquire spinlock
        while ct.atomic_cas(Locks, group_id, 0, 1;
                           memory_order=ct.MemoryOrder.Acquire) == 1
            # spin
        end

        # Critical section: accumulate partial gradients
        partial_dw = partial_dw .+ ct.load(DW, (i, group_id), (TILE_M, 1); padding_mode)
        partial_db = partial_db .+ ct.load(DB, (i, group_id), (TILE_M, 1); padding_mode)
        ct.store(DW, (i, group_id), partial_dw)
        ct.store(DB, (i, group_id), partial_db)

        # Release spinlock
        ct.atomic_xchg(Locks, group_id, 0;
                      memory_order=ct.MemoryOrder.Release)

        i += 1i32
    end

    return
end

"""
    layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, TILE_G, TILE_F)

Backward pass part 2: Final reduction for dW and dB.

Args:
    DW: Partial gradient with respect to W (M, N_GROUPS).
    DB: Partial gradient with respect to B (M, N_GROUPS).
    FINAL_DW: Final gradient with respect to W (M,).
    FINAL_DB: Final gradient with respect to B (M,).
    TILE_G: Tile size along groups dimension (dim 2).
    TILE_F: Tile size along feature dimension (dim 1).
"""
function layer_norm_bwd_dwdb(
    DW::TileMatrix{Float32}, DB::TileMatrix{Float32},
    FINAL_DW::TileVector{Float32}, FINAL_DB::TileVector{Float32},
    TILE_G::Int, TILE_F::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid = ct.bid(1)
    num_tiles = ct.num_tiles(DW, 2, (TILE_F, TILE_G))

    dw = ct.zeros((TILE_F, TILE_G), Float32)
    db = ct.zeros((TILE_F, TILE_G), Float32)
    i = 1i32
    while i <= num_tiles
        dw = dw .+ ct.load(DW, (bid, i), (TILE_F, TILE_G); padding_mode)
        db = db .+ ct.load(DB, (bid, i), (TILE_F, TILE_G); padding_mode)
        i += 1i32
    end
    sum_dw = sum(dw; dims=2)
    sum_db = sum(db; dims=2)

    ct.store(FINAL_DW, bid, sum_dw)
    ct.store(FINAL_DB, bid, sum_db)

    return
end

function layer_norm(
    X::AbstractMatrix{T}, W::AbstractVector{Tw}, B::AbstractVector{Tw};
    eps,
    verify = nothing
) where {T, Tw}
    M, N = size(X)
    @assert length(W) == length(B) == M

    Y = similar(X)
    Mean = similar(X, Float32, N)
    Rstd = similar(X, Float32, N)

    key = (T, Tw)

    autotune_launch(layer_norm_fwd,
        CartesianSpace(TILE_M=(128, 256, 512, 1024)),
        cfg -> N,
        cfg -> (
            X, W, B, Y, Mean, Rstd,
            Constant(Float32(eps)), Constant(cfg.TILE_M)
        );
        key, verify
    )

    return Y, Mean, Rstd
end

function ∇layer_norm(
    Ȳ::AbstractMatrix, X::AbstractMatrix,
    W::AbstractVector, B::AbstractVector,
    Mean::AbstractVector, Rstd::AbstractVector;
    N_GROUPS = 64, # XXX: autotune this. difficult because two kernel launches
    # XXX: verify
)
    M, N = size(X)

    X̄ = similar(X)
    W̄_partial = fill!(similar(W, M, N_GROUPS), 0)
    B̄_partial = fill!(similar(B, M, N_GROUPS), 0)
    Locks = fill!(similar(X, Int, N_GROUPS), 0)
    W̄ = similar(W)
    B̄ = similar(B)

    key = (eltype(X), eltype(W))

    autotune_launch(layer_norm_bwd_dx_partial_dwdb,
        CartesianSpace(TILE_M=(128, 256, 512, 1024)),
        cfg -> N,
        cfg -> (
            X̄, Ȳ, fill!(W̄_partial, 0), fill!(B̄_partial, 0), X, W,
            Mean, Rstd, fill!(Locks, 0),
            Constant(N_GROUPS),
            Constant(cfg.TILE_M)
        )
    )

    # TODO (cuTile.jl): result function to take args constructed from cfg?
    # (like {W̄,B̄}_partial with tuned N_GROUPS)
    autotune_launch(layer_norm_bwd_dwdb,
        CartesianSpace(TILE_F=(128, 256, 512, 1024), TILE_G=(32,)),
        cfg -> cld(M, cfg.TILE_F),
        cfg -> (
            W̄_partial, B̄_partial, W̄, B̄,
            Constant(cfg.TILE_G), Constant(cfg.TILE_F)
        );
        key,
    )

    return X̄, W̄, B̄
end
