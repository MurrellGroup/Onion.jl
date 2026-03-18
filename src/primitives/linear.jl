using LinearAlgebra: mul!

"""
    linear(x::AbstractMatrix, W::AbstractMatrix, b)

Matrix multiply with optional bias: `W * x .+ b`.
`b` can be an `AbstractVector` or `false` (no bias).
"""
@primitive linear
@primitive linear!

function linear!(::DefaultBackend,
    y::AbstractMatrix, x::AbstractMatrix, W::AbstractMatrix, b::Union{AbstractVector,Bool}
)
    mul!(y, W, x)
    NNlib.bias_act!(identity, y, b)
    return y
end

function linear(::DefaultBackend,
    x::AbstractMatrix, W::AbstractMatrix, b::Union{AbstractVector,Bool}
)
    y = W * x
    NNlib.bias_act!(identity, y, b)
    return y
end
