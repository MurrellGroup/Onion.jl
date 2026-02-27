# ── Gram kernel (O2): D = XᵀX (tall) or XXᵀ (wide), upper-triangle compute ──

function ns_gram_kernel(
    X::TileArray3,
    D::TileArray3{To},
    Tc::Type, Tacc::Type,
    TILE_N::Int, TILE_K::Int, GROUP_SIZE_M::Int,
    TALL::Bool,
) where To
    padding_mode = ct.PaddingMode.Zero
    bid, batch = ct.bid(1), ct.bid(2)

    N = size(D, 1)
    K = TALL ? size(X, 1) : size(X, 2)

    i_0, j_0 = swizzle_2d(N, N, TILE_N, TILE_N, GROUP_SIZE_M, bid - 1i32)
    i = i_0 + 1i32
    j = j_0 + 1i32

    # Upper triangle only: skip blocks where j < i
    j >= i || return

    acc = ct.zeros((TILE_N, TILE_N), Tacc)
    num_k = cld(K, Int32(TILE_K))

    k = 1i32
    while k <= num_k
        if TALL
            # XᵀX: X is (M, N, B), contract over M
            x_i = reshape(ct.load(X, (k, i, batch), (TILE_K, TILE_N, 1); padding_mode), (TILE_K, TILE_N))
            x_j = reshape(ct.load(X, (k, j, batch), (TILE_K, TILE_N, 1); padding_mode), (TILE_K, TILE_N))
            acc = muladd((x_i)ᵀ → Tc, x_j → Tc, acc)
        else
            # XXᵀ: X is (M, N, B), contract over N
            x_i = reshape(ct.load(X, (i, k, batch), (TILE_N, TILE_K, 1); padding_mode), (TILE_N, TILE_K))
            x_j = reshape(ct.load(X, (j, k, batch), (TILE_N, TILE_K, 1); padding_mode), (TILE_N, TILE_K))
            acc = muladd(x_i → Tc, (x_j)ᵀ → Tc, acc)
        end
        k += 1i32
    end

    d = acc → To
    ct.store(D, (i, j, batch), reshape(d, (TILE_N, TILE_N, 1)))
    # Mirror to lower triangle
    if i != j
        ct.store(D, (j, i, batch), reshape(transpose(d), (TILE_N, TILE_N, 1)))
    end

    return
end

function ns_gram!(
    D::AbstractArray{To,3}, X::AbstractArray{<:Any,3};
    tall::Bool,
    tensorcore = tensorcore_type(eltype(X)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing,
) where To
    N = size(D, 1)
    K = tall ? size(X, 1) : size(X, 2)
    B = size(X, 3)
    @assert size(D) == (N, N, B)

    key = (eltype(X), tensorcore, accumulate, nextpow.(4, (N, K)), tall)

    autotune_launch(ns_gram_kernel,
        CartesianSpace(
            TILE_N=(32, 64, 128), TILE_K=(32, 64, 128),
            GROUP_SIZE_M=(8, 16), occupancy=(1, 2, 4),
        ),
        cfg -> (cld(N, cfg.TILE_N)^2, B),
        cfg -> (
            X, D,
            Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_N), Constant(cfg.TILE_K),
            Constant(cfg.GROUP_SIZE_M),
            Constant(tall),
        );
        key, verify,
    )
    return D
end

# ── Symmetric GEMM kernel (O5): D = α·A·B + β·C, upper-triangle output ──
# For symmetric A·B where both A and B are symmetric (so A·B is symmetric).
# Only computes tiles (i, j) with j >= i, mirrors to lower triangle.

function ns_sym_gemm_kernel(
    A::TileArray3, B::TileArray3,
    C::Optional{TileArray3},
    D::TileArray3{To},
    α::Float32, β::Float32,
    Tc::Type, Tacc::Type,
    TILE_N::Int, TILE_K::Int, GROUP_SIZE_M::Int,
) where To
    padding_mode = ct.PaddingMode.Zero
    bid, batch = ct.bid(1), ct.bid(2)

    N = size(D, 1)
    K = size(A, 2)

    i_0, j_0 = swizzle_2d(N, N, TILE_N, TILE_N, GROUP_SIZE_M, bid - 1i32)
    i = i_0 + 1i32
    j = j_0 + 1i32

    # Upper triangle only
    j >= i || return

    acc = ct.zeros((TILE_N, TILE_N), Tacc)
    num_k = cld(K, Int32(TILE_K))

    k = 1i32
    while k <= num_k
        a = reshape(ct.load(A, (i, k, batch), (TILE_N, TILE_K, 1); padding_mode), (TILE_N, TILE_K))
        b = reshape(ct.load(B, (k, j, batch), (TILE_K, TILE_N, 1); padding_mode), (TILE_K, TILE_N))
        acc = muladd(a → Tc, b → Tc, acc)
        k += 1i32
    end

    d = if isnothing(C)
        α .* acc
    else
        c = reshape(ct.load(C, (i, j, batch), (TILE_N, TILE_N, 1); padding_mode), (TILE_N, TILE_N)) → Tacc
        fma.(c, Tacc(β), Tacc(α) .* acc)
    end

    ct.store(D, (i, j, batch), reshape(d, (TILE_N, TILE_N, 1)) → To)
    # Mirror to lower triangle
    if i != j
        ct.store(D, (j, i, batch), reshape(transpose(d), (TILE_N, TILE_N, 1)) → To)
    end

    return
end

function ns_sym_gemm!(
    D::AbstractArray{To,3},
    A::AbstractArray{<:Any,3}, B::AbstractArray{<:Any,3},
    C::Optional{AbstractArray{<:Any,3}} = nothing,
    α::Float32 = 1f0, β::Float32 = 0f0;
    tensorcore = tensorcore_type(eltype(A)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing,
) where To
    N = size(D, 1)
    K = size(A, 2)
    Batch = size(D, 3)
    @assert size(D) == (N, N, Batch)
    @assert size(A, 1) == N && size(A, 3) == Batch
    @assert size(B) == (K, N, Batch)
    isnothing(C) || @assert size(C) == (N, N, Batch)

    key = (eltype(A), tensorcore, accumulate, nextpow.(4, (N, K)), !isnothing(C))

    autotune_launch(ns_sym_gemm_kernel,
        CartesianSpace(
            TILE_N=(32, 64, 128), TILE_K=(32, 64, 128),
            GROUP_SIZE_M=(8, 16), occupancy=(1, 2, 4),
        ),
        cfg -> (cld(N, cfg.TILE_N)^2, Batch),
        cfg -> (
            A, B, C, D,
            α, β,
            Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_N), Constant(cfg.TILE_K),
            Constant(cfg.GROUP_SIZE_M),
        );
        key, verify,
    )
    return D
end

# ── GEMM kernel (O1): D = α·A·B + β·C, with C ≠ D ──

function ns_gemm_kernel(
    A::TileArray3, B::TileArray3,
    C::Optional{TileArray3},
    D::TileArray3{To},
    α::Float32, β::Float32,
    Tc::Type, Tacc::Type,
    TILE_M::Int, TILE_N::Int, TILE_K::Int, GROUP_SIZE_M::Int,
) where To
    padding_mode = ct.PaddingMode.Zero
    bid, batch = ct.bid(1), ct.bid(2)

    M, N = size(D, 1), size(D, 2)
    K = size(A, 2)

    i_0, j_0 = swizzle_2d(M, N, TILE_M, TILE_N, GROUP_SIZE_M, bid - 1i32)
    i = i_0 + 1i32
    j = j_0 + 1i32

    acc = ct.zeros((TILE_M, TILE_N), Tacc)
    num_k = cld(K, Int32(TILE_K))

    k = 1i32
    while k <= num_k
        a = reshape(ct.load(A, (i, k, batch), (TILE_M, TILE_K, 1); padding_mode), (TILE_M, TILE_K))
        b = reshape(ct.load(B, (k, j, batch), (TILE_K, TILE_N, 1); padding_mode), (TILE_K, TILE_N))
        acc = muladd(a → Tc, b → Tc, acc)
        k += 1i32
    end

    d = if isnothing(C)
        α .* acc
    else
        c = reshape(ct.load(C, (i, j, batch), (TILE_M, TILE_N, 1); padding_mode), (TILE_M, TILE_N)) → Tacc
        fma.(c, Tacc(β), Tacc(α) .* acc)
    end

    ct.store(D, (i, j, batch), reshape(d, (TILE_M, TILE_N, 1)) → To)
    return
end

function ns_gemm!(
    D::AbstractArray{To,3},
    A::AbstractArray{<:Any,3}, B::AbstractArray{<:Any,3},
    C::Optional{AbstractArray{<:Any,3}} = nothing,
    α::Float32 = 1f0, β::Float32 = 0f0;
    tensorcore = tensorcore_type(eltype(A)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing,
) where To
    M, N = size(D, 1), size(D, 2)
    K = size(A, 2)
    Batch = size(D, 3)
    @assert size(A, 1) == M && size(A, 3) == Batch
    @assert size(B) == (K, N, Batch)
    isnothing(C) || @assert size(C) == (M, N, Batch)

    key = (eltype(A), tensorcore, accumulate, nextpow.(4, (M, K, N)), !isnothing(C))

    autotune_launch(ns_gemm_kernel,
        CartesianSpace(
            TILE_M=(32, 64, 128), TILE_K=(32, 64, 128),
            TILE_N=(32, 64, 128), GROUP_SIZE_M=(8, 16),
            occupancy=(1, 2, 4),
        ),
        cfg -> (cld(M, cfg.TILE_M) * cld(N, cfg.TILE_N), Batch),
        cfg -> (
            A, B, C, D,
            α, β,
            Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_M), Constant(cfg.TILE_N), Constant(cfg.TILE_K),
            Constant(cfg.GROUP_SIZE_M),
        );
        key, verify,
    )
    return D
end

# ── Newton-Schulz iteration orchestrator ──

function newton_schulz!(
    X::AbstractArray{T,3}, coefficients;
    arithmetic::Type{Ta} = BFloat16,
    tensorcore = tensorcore_type(arithmetic),
    accumulate = accumulate_type(tensorcore),
) where {T, Ta}
    M, N, B = size(X)
    tall = M > N
    n = tall ? N : M

    # Pre-allocate buffers in arithmetic type (BFloat16 by default)
    A = similar(X, Ta, n, n, B)
    S = similar(X, Ta, n, n, B)
    Y = similar(X, Ta, M, N, B)
    X_cur = Ta.(X)

    tc_kws = (; tensorcore, accumulate)

    for (a, b, c) in coefficients
        # ① Gram: A = XᵀX or XXᵀ
        ns_gram!(A, X_cur; tall, tc_kws...)

        # ② Polynomial: S = c·A² + b·A  (symmetric — A is symmetric, so A² is too)
        ns_sym_gemm!(S, A, A, A, Float32(c), Float32(b); tc_kws...)

        # ③ Final: Y = 1·X·S + a·X or Y = 1·S·X + a·X
        if tall
            ns_gemm!(Y, X_cur, S, X_cur, 1f0, Float32(a); tc_kws...)
        else
            ns_gemm!(Y, S, X_cur, X_cur, 1f0, Float32(a); tc_kws...)
        end

        X_cur, Y = Y, X_cur
    end

    return T.(X_cur)
end
