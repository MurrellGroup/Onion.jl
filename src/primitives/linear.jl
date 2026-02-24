function linear(::DefaultBackend,
    x::AbstractMatrix, W::AbstractMatrix, b::Union{AbstractVector,Bool}
)
    y = W * x
    NNlib.bias_act!(identity, y, b)
    return y
end
