using NNlib: batched_mul!, batched_transpose

"""
    newton_schulz(X, coefficients)

Quintic Newton-Schulz iteration for polar decomposition.
`coefficients` is an iterable of `(a, b, c)` tuples — one per iteration.
Each step applies `Y = aX + bXXᵀX + cXXᵀXXᵀX` (tall) or the wide variant.
"""
@primitive _newton_schulz as newton_schulz
@primitive _newton_schulz! as newton_schulz!

function _newton_schulz!(::DefaultBackend,
    out::AbstractArray{T}, X::AbstractArray{T}, coefficients
) where T
    result = _newton_schulz(DefaultBackend(), X, coefficients)
    if ndims(result) != ndims(out)
        result = reshape(result, size(out))
    end
    copyto!(out, result)
    return out
end

function _newton_schulz(::DefaultBackend,
    X::AbstractArray{T}, coefficients,
) where T
    input_is_matrix = ndims(X) == 2
    X3 = input_is_matrix ? reshape(X, size(X, 1), size(X, 2), 1) : X

    M, N, B = size(X3)
    tall = M > N
    n = tall ? N : M

    A = similar(X3, T, n, n, B)
    S = similar(X3, T, n, n, B)
    Y = similar(X3, T, M, N, B)
    X_cur = copy(X3)

    for (a, b, c) in coefficients
        Xt = batched_transpose(X_cur)

        if tall
            batched_mul!(A, Xt, X_cur)
            copyto!(S, A)
            batched_mul!(S, A, A, T(c), T(b))
            copyto!(Y, X_cur)
            batched_mul!(Y, X_cur, S, one(T), T(a))
        else
            batched_mul!(A, X_cur, Xt)
            copyto!(S, A)
            batched_mul!(S, A, A, T(c), T(b))
            copyto!(Y, X_cur)
            batched_mul!(Y, S, X_cur, one(T), T(a))
        end

        X_cur, Y = Y, X_cur
    end

    result = input_is_matrix ? reshape(X_cur, M, N) : X_cur
    return result
end
