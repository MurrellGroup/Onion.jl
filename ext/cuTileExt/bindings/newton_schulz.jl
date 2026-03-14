function Onion._newton_schulz(::cuTileBackend,
    X::AbstractArray{T}, coefficients,
) where T
    input_is_matrix = ndims(X) == 2
    X3 = input_is_matrix ? reshape(X, size(X, 1), size(X, 2), 1) : X

    result = newton_schulz!(X3, coefficients)

    return input_is_matrix ? reshape(result, size(X, 1), size(X, 2)) : result
end
