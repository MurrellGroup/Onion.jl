using NNlib: NNlib

"""
    softmax(x::AbstractMatrix)

Softmax on dim 1. Input must be 2D — callers reshape if needed.
"""
@primitive softmax
@primitive softmax!

function softmax!(::DefaultBackend, y::AbstractMatrix, x::AbstractMatrix)
    NNlib.softmax!(y, x; dims=1)
    return y
end

get_buffers(::typeof(softmax), b::Backend, x) = (; y = similar(x))

function softmax(::DefaultBackend, x::AbstractMatrix)
    return NNlib.softmax(x; dims=1)
end
