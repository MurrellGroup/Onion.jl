using Statistics: mean, var

"""
    layer_norm(x::AbstractMatrix, w::AbstractVector, b::AbstractVector; eps)

Layer normalization on dim 1. Input must be 2D — callers reshape if needed.
"""
@primitive _layer_norm as layer_norm
@primitive _layer_norm! as layer_norm!

function _layer_norm!(::DefaultBackend,
    y::AbstractMatrix, x::AbstractMatrix, w::AbstractVector, b::AbstractVector;
    eps
)
    μ = mean(x; dims=1)
    σ² = var(x; dims=1, mean=μ, corrected=false)
    y .= (x .- μ) ./ sqrt.(σ² .+ eps) .* w .+ b
    return y
end

function _layer_norm(::DefaultBackend,
    x::AbstractMatrix, w::AbstractVector, b::AbstractVector;
    eps
)
    μ = mean(x; dims=1)
    σ² = var(x; dims=1, mean=μ, corrected=false)
    y = (x .- μ) ./ sqrt.(σ² .+ eps) .* w .+ b
    return y
end
