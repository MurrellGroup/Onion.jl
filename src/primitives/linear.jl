using LinearAlgebra: mul!

"""
    linear(x::AbstractMatrix, W::AbstractMatrix, b)

Matrix multiply with optional bias: `W * x .+ b`.
`b` can be an `AbstractVector` or `nothing` (no bias).
"""
@primitive linear
@primitive linear!

function linear!(::DefaultBackend,
    y::AbstractMatrix, x::AbstractMatrix, W::AbstractMatrix, b::Optional{AbstractVector}
)
    if isnothing(b)
        mul!(y, W, x)
    else
        mul!(y .= b, W, x, true, true)
    end
    NNlib.bias_act!(identity, y, @something b false)
    return y
end

get_buffers(::typeof(linear), b::Backend, x, W, bias) =
    (; y = similar(x, size(W, 1), size(x, 2)))

function linear(backend::Backend,
    x::AbstractMatrix, W::AbstractMatrix, b::Optional{AbstractVector} = nothing
)
    bufs = get_buffers(linear, backend, x, W, b)
    return linear!(backend, bufs.y, x, W, b)
end

function linear(::DefaultBackend,
    x::AbstractMatrix, W::AbstractMatrix, b::Optional{AbstractVector} = nothing
)
    y = NNlib.bias_act!(identity, W * x, @something b false)
    return y
end
