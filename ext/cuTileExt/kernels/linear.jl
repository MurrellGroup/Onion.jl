function linear_kernel(
    W::TileMatrix, B::Optional{TileVector},
    X::TileMatrix, Y::TileMatrix,
    Tc::Type, Tacc::Type,
    TILE_M::Int, TILE_N::Int, TILE_K::Int,
    GROUP_SIZE_M::Int,
    W_TRANS::Bool, X_TRANS::Bool,
)
    padding_mode = ct.PaddingMode.Zero
    bid = ct.bid(1)
    M, N = size(Y)

    bid_m_0, bid_n_0 = swizzle_2d(M, N, TILE_M, TILE_N, GROUP_SIZE_M, bid - 1i32)
    bid_m = bid_m_0 + 1i32
    bid_n = bid_n_0 + 1i32

    num_k = cld(size(W, W_TRANS ? 1 : 2), Int32(TILE_K))

    acc = ct.zeros((TILE_M, TILE_N), Tacc)

    w_order = W_TRANS ? (2, 1) : (1, 2)
    x_order = X_TRANS ? (2, 1) : (1, 2)

    k = 1i32
    while k <= num_k
        w = ct.load(W, (bid_m, k), (TILE_M, TILE_K); padding_mode, order=w_order)
        x = ct.load(X, (k, bid_n), (TILE_K, TILE_N); padding_mode, order=x_order)
        acc = muladd(w → Tc, x → Tc, acc)
        k += 1i32
    end

    if !isnothing(B)
        b = ct.load(B, (bid_m,), (TILE_M,))
        acc = acc .+ b → Tacc
    end

    ct.store(Y, (bid_m, bid_n), acc → eltype(Y))
    return
end

function linear!(
    W::AbstractMatrix, B::Optional{AbstractVector},
    X::AbstractMatrix, Y::AbstractMatrix;
    transpose_w::Bool = false,
    transpose_x::Bool = false,
    tensorcore = tensorcore_type(eltype(X)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing,
)
    M, N = size(Y)
    K = transpose_w ? size(W, 1) : size(W, 2)
    @assert (transpose_w ? size(W, 2) : size(W, 1)) == M
    @assert (transpose_x ? size(X, 1) : size(X, 2)) == N
    @assert (transpose_x ? size(X, 2) : size(X, 1)) == K
    isnothing(B) || @assert size(B) == (M,)
    key = (eltype(X), tensorcore, accumulate, nextpow.(4, (M, K, N)),
           !isnothing(B), transpose_w, transpose_x)
    autotune_launch(linear_kernel,
        CartesianSpace(
            TILE_M=(32, 64, 128), TILE_K=(32, 64, 128),
            TILE_N=(32, 64, 128), GROUP_SIZE_M=(8, 16),
            occupancy=(1, 2, 4),
        ),
        cfg -> (cld(M, cfg.TILE_M) * cld(N, cfg.TILE_N),),
        cfg -> (
            W, B, X, Y,
            Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_M), Constant(cfg.TILE_N), Constant(cfg.TILE_K),
            Constant(cfg.GROUP_SIZE_M),
            Constant(transpose_w), Constant(transpose_x),
        );
        key, verify,
    )
    return nothing
end

function ∇linear!(
    X̄::AbstractMatrix, W̄::AbstractMatrix, B̄::Optional{AbstractVector},
    Ȳ::AbstractMatrix,
    W::AbstractMatrix, X::AbstractMatrix;
    tensorcore = tensorcore_type(eltype(X)),
    accumulate = accumulate_type(tensorcore),
    verify_x = nothing,
    verify_w = nothing,
)
    # X̄ = Wᵀ @ Ȳ
    linear!(W, nothing, Ȳ, X̄;
        transpose_w=true, tensorcore, accumulate, verify=verify_x)

    # W̄ = Ȳ @ Xᵀ
    linear!(Ȳ, nothing, X, W̄;
        transpose_x=true, tensorcore, accumulate, verify=verify_w)

    # B̄ = sum(Ȳ, dims=2)
    if !isnothing(B̄)
        B̄ .= sum(Ȳ; dims=2)
    end

    return
end
