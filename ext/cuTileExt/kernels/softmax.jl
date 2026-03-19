function softmax_fwd(
    X::TileMatrix{Float32}, Y::TileMatrix{Float32},
    TILE_M::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)

    # Pass 1 (online): compute running max and sum(exp(x - max))
    m = fill(-Inf32, TILE_M)
    s = zeros(Float32, TILE_M)
    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,); padding_mode)
        # Mask padded elements to -Inf
        mask = ((i - 1i32) * Int32(TILE_M) .+ ct.arange(TILE_M, Int32)) .<= M
        x = ifelse.(mask, x, -Inf32)

        m_new = max.(m, x)
        safe = m_new .> -Inf32
        s = s .* ifelse.(safe, exp.(m .- m_new), 0.0f0) .+ ifelse.(safe, exp.(x .- m_new), 0.0f0)
        m = m_new
    end

    m_global = maximum(m)
    s_global = sum(s .* exp.(m .- m_global))

    # Pass 2: normalize
    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,); padding_mode)
        mask = ((i - 1i32) * Int32(TILE_M) .+ ct.arange(TILE_M, Int32)) .<= M
        x = ifelse.(mask, x, -Inf32)
        y = exp.(x .- m_global) ./ s_global
        ct.store(Y, (i, bid_n), y)
    end

    return
end

function softmax_bwd(
    X̄::TileMatrix{Float32}, Ȳ::TileMatrix{Float32},
    Y::TileMatrix{Float32},
    TILE_M::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(Y, 1, (TILE_M, 1))

    acc = zeros(Float32, TILE_M)
    for i in 1i32:num_tiles
        y = ct.load(Y, (i, bid_n), (TILE_M,); padding_mode)
        ȳ = ct.load(Ȳ, (i, bid_n), (TILE_M,); padding_mode)
        acc = acc .+ (y .* ȳ)
    end
    dot = sum(acc)

    for i in 1i32:num_tiles
        y = ct.load(Y, (i, bid_n), (TILE_M,); padding_mode)
        ȳ = ct.load(Ȳ, (i, bid_n), (TILE_M,); padding_mode)
        x̄ = y .* (ȳ .- dot)
        ct.store(X̄, (i, bid_n), x̄)
    end

    return
end

function softmax!(Y::AbstractMatrix, X::AbstractMatrix{T}; verify = nothing)
    autotune_launch(softmax_fwd,
        CartesianSpace(TILE_M=(128, 256, 512, 1024)),
        cfg -> size(X, 2),
        cfg -> (X, Y, Constant(cfg.TILE_M));
        key = eltype(X), verify
    )

    return
end

function ∇softmax(Ȳ::AbstractMatrix, Y::AbstractMatrix)
    X̄ = similar(X)

    autotune_launch(softmax_bwd,
        CartesianSpace(TILE_M=(128, 256, 512, 1024)),
        cfg -> size(Y, 2),
        cfg -> (X̄, Ȳ, Y, Constant(cfg.TILE_M));
        key = eltype(Y),
    )

    return X̄
end
