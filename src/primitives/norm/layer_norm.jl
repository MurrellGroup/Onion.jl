using Statistics: mean, var

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
