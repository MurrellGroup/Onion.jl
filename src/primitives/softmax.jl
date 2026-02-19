using NNlib: NNlib

function softmax(::DefaultBackend,
    x::AbstractArray, dims::Int = 1
)
    return NNlib.softmax(x; dims)
end
