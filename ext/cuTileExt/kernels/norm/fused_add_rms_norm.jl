# Fused: new_x = residual + x; normed = (new_x / rms) * (w + offset)
# Single kernel, two outputs. Column-major (M, N), normalizes along dim 1.
# Loads as Float32 for precision, stores as OUT_T (BFloat16 or Float32).

function fused_add_rms_norm_fwd(
    Residual::TileMatrix, X::TileMatrix, W::TileVector,
    NewX::TileMatrix, Normed::TileMatrix,
    offset::Float32, eps::Float32, TILE_M::Int, OUT_T::Type,
)
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)

    # Pass 1: add + sum of squares
    ss = ct.full((1, TILE_M), 0.0f0, Float32)
    i = 1i32
    while i <= num_tiles
        tr = reshape(ct.load(Residual, (i, bid_n), (TILE_M, 1); padding_mode=ct.PaddingMode.Zero) → Float32, (1, TILE_M))
        tx = reshape(ct.load(X, (i, bid_n), (TILE_M, 1); padding_mode=ct.PaddingMode.Zero) → Float32, (1, TILE_M))
        tnew = tr .+ tx
        ct.store(NewX, (i, bid_n), reshape(tnew → OUT_T, (TILE_M, 1)))
        ss = ss .+ (tnew .* tnew)
        i += 1i32
    end
    rstd = 1.0f0 ./ sqrt.(sum(ss; dims=2) / M .+ eps)

    # Pass 2: normalize with weight
    i = 1i32
    while i <= num_tiles
        tnew = reshape(ct.load(NewX, (i, bid_n), (TILE_M, 1); padding_mode=ct.PaddingMode.Zero) → Float32, (1, TILE_M))
        tw = reshape(ct.load(W, i, (TILE_M,); padding_mode=ct.PaddingMode.Zero) → Float32, (1, TILE_M))
        ty = tnew .* rstd .* (tw .+ offset)
        ct.store(Normed, (i, bid_n), reshape(ty → OUT_T, (TILE_M, 1)))
        i += 1i32
    end

    return
end

function fused_add_rms_norm_step!(NewX, Normed, Residual, X, W; eps, offset, verify=nothing)
    M, N = size(X)
    T = eltype(X)
    key = (T, M)

    autotune_launch(fused_add_rms_norm_fwd,
        CartesianSpace(TILE_M=(128, 256, 512, 1024)),
        cfg -> N,
        cfg -> (
            Residual, X, W, NewX, Normed,
            Constant(Float32(offset)), Constant(Float32(eps)),
            Constant(cfg.TILE_M), Constant(T)
        );
        key, verify
    )
end
