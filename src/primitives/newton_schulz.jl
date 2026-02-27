using NNlib: batched_mul!, batched_transpose

function newton_schulz(::DefaultBackend,
    X::AbstractArray{T}, coefficients,
) where T
    input_is_matrix = ndims(X) == 2
    X3 = input_is_matrix ? reshape(X, size(X, 1), size(X, 2), 1) : X

    M, N, B = size(X3)
    tall = M > N
    n = tall ? N : M

    # Pre-allocate: square buffers + output buffer
    A = similar(X3, T, n, n, B)
    S = similar(X3, T, n, n, B)
    Y = similar(X3, T, M, N, B)
    X_cur = copy(X3)

    for (a, b, c) in coefficients
        Xt = batched_transpose(X_cur)

        if tall
            batched_mul!(A, Xt, X_cur)              # A = XᵀX
            copyto!(S, A)
            batched_mul!(S, A, A, T(c), T(b))       # S = c·A² + b·A
            copyto!(Y, X_cur)
            batched_mul!(Y, X_cur, S, one(T), T(a)) # Y = X·S + a·X
        else
            batched_mul!(A, X_cur, Xt)              # A = XXᵀ
            copyto!(S, A)
            batched_mul!(S, A, A, T(c), T(b))       # S = c·A² + b·A
            copyto!(Y, X_cur)
            batched_mul!(Y, S, X_cur, one(T), T(a)) # Y = S·X + a·X
        end

        X_cur, Y = Y, X_cur
    end

    result = input_is_matrix ? reshape(X_cur, M, N) : X_cur
    return result
end
