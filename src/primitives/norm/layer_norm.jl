using Statistics: mean, var

"""
    layer_norm(x::AbstractMatrix, w::AbstractVector, b::AbstractVector; eps)
"""
@primitive _layer_norm as layer_norm
@primitive _layer_norm! as layer_norm!

function _layer_norm!(::DefaultBackend,
    y::AbstractArray, x::AbstractArray, w::AbstractVector, b::AbstractVector,
    dims::Union{Int,Val{1}};
    eps
)
    dims_val = unval(dims)
    μ = mean(x; dims=dims_val)
    σ² = var(x; dims=dims_val, mean=μ, corrected=false)
    y .= (x .- μ) ./ sqrt.(σ² .+ eps) .* w .+ b
    return y
end

function _layer_norm(::DefaultBackend,
    x::AbstractArray, w::AbstractVector, b::AbstractVector,
    dims::Union{Int,Val{1}};
    eps
)
    dims = unval(dims)
    μ = mean(x; dims)
    σ² = var(x; dims, mean=μ, corrected=false)
    y = (x .- μ) ./ sqrt.(σ² .+ eps) .* w .+ b
    return y
end

function layer_norm(backend::Backend,
    x::AbstractArray, w::AbstractVector, b::AbstractVector;
    dims::Union{Int,Val{1}} = Val(1),
    eps
)
    if dims isa Val{1}
        x′ = reshape(x, Keep(), :)
        y′ = _layer_norm(backend, x′, w, b, dims; eps)
        return reshape(y′, Keep(), Split(.., size(x)[2:end]))
    end
    return _layer_norm(backend, x, w, b, dims; eps)
end

function layer_norm!(backend::Backend,
    y::AbstractArray, x::AbstractArray, w::AbstractVector, b::AbstractVector;
    dims::Union{Int,Val{1}} = Val(1),
    eps
)
    if dims isa Val{1}
        x′ = reshape(x, Keep(), :)
        y′ = reshape(y, Keep(), :)
        _layer_norm!(backend, y′, x′, w, b, dims; eps)
        return y
    end
    return _layer_norm!(backend, y, x, w, b, dims; eps)
end
