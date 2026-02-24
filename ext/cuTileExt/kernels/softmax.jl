function softmax_fwd(
    X::TileMatrix{Float32}, Y::TileMatrix{Float32},
    TILE_M::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)

    # Pass 1 (online): compute running max and sum(exp(x - max))
    m = ct.full((1, TILE_M), -Inf32, Float32)
    s = ct.full((1, TILE_M), 0.0f0, Float32)
    i = 1i32
    while i <= num_tiles
        tx = reshape(ct.load(X, (i, bid_n), (TILE_M, 1); padding_mode), (1, TILE_M))
        # Mask padded elements to -Inf
        mask = reshape(((i - 1i32) * Int32(TILE_M) .+ ct.arange((TILE_M,), Int32)) .<= M, (1, TILE_M))
        tx = ifelse.(mask, tx, -Inf32)

        # Online softmax update
        m_new = max.(m, tx)
        safe = m_new .> -Inf32
        s = s .* ifelse.(safe, exp.(m .- m_new), 0.0f0) .+ ifelse.(safe, exp.(tx .- m_new), 0.0f0)
        m = m_new
        i += 1i32
    end

    # Reduce across TILE_M elements to get global max and sum
    m_global = maximum(m; dims=2)
    s_global = sum(s .* exp.(m .- m_global); dims=2)

    # Pass 2: normalize
    i = 1i32
    while i <= num_tiles
        tx = reshape(ct.load(X, (i, bid_n), (TILE_M, 1); padding_mode), (1, TILE_M))
        mask = reshape(((i - 1i32) * Int32(TILE_M) .+ ct.arange((TILE_M,), Int32)) .<= M, (1, TILE_M))
        tx = ifelse.(mask, tx, -Inf32)
        ty = exp.(tx .- m_global) ./ s_global
        ct.store(Y, (i, bid_n), reshape(ty, (TILE_M, 1)))
        i += 1i32
    end

    return
end

function softmax_bwd(
    DX::TileMatrix{Float32}, DY::TileMatrix{Float32},
    Y::TileMatrix{Float32},
    TILE_M::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(Y, 1, (TILE_M, 1))

    # Pass 1: compute dot = sum(y * dy)
    dot = ct.full((1, TILE_M), 0.0f0, Float32)
    i = 1i32
    while i <= num_tiles
        ty = reshape(ct.load(Y, (i, bid_n), (TILE_M, 1); padding_mode), (1, TILE_M))
        tdy = reshape(ct.load(DY, (i, bid_n), (TILE_M, 1); padding_mode), (1, TILE_M))
        dot = dot .+ (ty .* tdy)
        i += 1i32
    end
    dot = sum(dot; dims=2)

    # Pass 2: dx = y * (dy - dot)
    i = 1i32
    while i <= num_tiles
        ty = reshape(ct.load(Y, (i, bid_n), (TILE_M, 1); padding_mode), (1, TILE_M))
        tdy = reshape(ct.load(DY, (i, bid_n), (TILE_M, 1); padding_mode), (1, TILE_M))
        tdx = ty .* (tdy .- dot)
        ct.store(DX, (i, bid_n), reshape(tdx, (TILE_M, 1)))
        i += 1i32
    end

    return
end

function online_softmax(X::AbstractMatrix{T}; verify = nothing) where {T}
    M, N = size(X)

    Y = similar(X)

    autotune_launch(softmax_fwd,
        CartesianSpace(TILE_M=(128, 256, 512, 1024)),
        cfg -> N,
        cfg -> (
            X, Y,
            Constant(cfg.TILE_M)
        );
        key = T, verify
    )

    return Y
end

function ∇online_softmax(
    Ȳ::AbstractMatrix, Y::AbstractMatrix)
    M, N = size(Y)

    X̄ = similar(Y)

    autotune_launch(softmax_bwd,
        CartesianSpace(TILE_M=(128, 256, 512, 1024)),
        cfg -> N,
        cfg -> (
            X̄, Ȳ, Y,
            Constant(cfg.TILE_M)
        );
        key = eltype(Y),
    )

    return X̄
end
