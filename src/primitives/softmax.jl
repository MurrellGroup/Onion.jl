using NNlib: NNlib

function _softmax(::DefaultBackend,
    x::AbstractArray, dims::Union{Int,Val{1}}
)
    return NNlib.softmax(x; dims=unval(dims))
end

function softmax(b::Backend,
    x::AbstractArray;
    dims::Union{Int,Val{1}} = Val(1)
)
    if dims isa Val{1}
        x′ = reshape(x, Keep(), :)
        y′ = _softmax(b, x′, dims)
        return reshape(y′, Keep(), Split(.., size(x)[2:end]))
    end
    return _softmax(b, x, dims)
end