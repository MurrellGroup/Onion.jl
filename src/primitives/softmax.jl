using NNlib: NNlib

@impl DefaultBackend function softmax(x::AbstractArray, dims::Int)
    return NNlib.softmax(x; dims)
end
