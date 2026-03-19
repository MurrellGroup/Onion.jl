# Fused: new_x = residual + x; normed = (new_x / rms) * (w + offset)
# Single kernel, two outputs. Column-major (M, N), normalizes along dim 1.

function fused_add_rms_norm_fwd(
    Residual::TileMatrix, X::TileMatrix, W::TileVector,
    NewX::TileMatrix, Normed::TileMatrix,
    offset::Float32, eps::Float32, TILE_M::Int,
)
    padding_mode = ct.PaddingMode.Zero
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)

    ss = ct.zeros((TILE_M,), Float32)
    for i in 1i32:num_tiles
        r = ct.load(Residual, (i, bid_n), (TILE_M,); padding_mode) → Float32
        x = ct.load(X, (i, bid_n), (TILE_M,); padding_mode) → Float32
        new = r .+ x
        ct.store(NewX, (i, bid_n), new → eltype(NewX))
        ss = ss .+ (new .* new)
    end
    rstd = 1 / √(sum(ss) / M + eps)

    for i in 1i32:num_tiles
        new = ct.load(NewX, (i, bid_n), (TILE_M,); padding_mode) → Float32
        w = ct.load(W, i, (TILE_M,); padding_mode) → Float32
        y = new .* rstd .* (w .+ offset)
        ct.store(Normed, (i, bid_n), y → eltype(Normed))
    end

    return
end

function fused_add_rms_norm!(NewX, Normed, Residual, X, W; eps, offset, verify=nothing)

    autotune_launch(fused_add_rms_norm_fwd,
        CartesianSpace(TILE_M=(128, 256, 512, 1024)),
        cfg -> size(X, 2),
        cfg -> (
            Residual, X, W, NewX, Normed,
            Constant(Float32(offset)), Constant(Float32(eps)),
            Constant(cfg.TILE_M)
        );
        key = (eltype(X), size(X, 1)),
        verify
    )

    return
end
