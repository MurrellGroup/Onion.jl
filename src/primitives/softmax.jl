using NNlib: NNlib

"""
    softmax(x::AbstractMatrix)

Softmax on dim 1. Input must be 2D — callers reshape if needed.
"""
@primitive _softmax as softmax
@primitive _softmax! as softmax!

function _softmax!(::DefaultBackend, y::AbstractMatrix, x::AbstractMatrix)
    NNlib.softmax!(y, x; dims=1)
    return y
end

function _softmax(::DefaultBackend, x::AbstractMatrix)
    return NNlib.softmax(x; dims=1)
end
