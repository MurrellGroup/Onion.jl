using NNlib: NNlib

"""
    softmax(x::AbstractMatrix)

Softmax over dimension 1. Input must be 2D — callers reshape if needed.
"""
@primitive softmax
@primitive softmax!

get_buffers(::typeof(softmax), b::Backend, x) = (; y = similar(x))

function softmax(b::Backend, x::AbstractMatrix)
    bufs = get_buffers(softmax, b, x)
    return softmax!(b, bufs.y, x)
end

function softmax(::DefaultBackend, x::AbstractMatrix)
    return NNlib.softmax(x; dims=1)
end
