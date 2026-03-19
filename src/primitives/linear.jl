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

get_buffers(::typeof(linear), b::Backend, x, W, bias) =
    (; y = similar(x, size(W, 1), size(x, 2)))

function linear(b::Backend,
    x::AbstractMatrix, W::AbstractMatrix, bias::Union{AbstractVector,Bool}
)
    bufs = get_buffers(linear, b, x, W, bias)
    return linear!(b, bufs.y, x, W, bias)
end
