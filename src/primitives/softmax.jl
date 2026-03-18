using NNlib: NNlib

"""
    softmax(x::AbstractMatrix; dims=1)
"""
@primitive _softmax as softmax
@primitive _softmax! as softmax!

function _softmax!(::DefaultBackend,
    y::AbstractArray, x::AbstractArray, dims::Union{Int,Val{1}}
)
    NNlib.softmax!(y, x; dims=unval(dims))
    return y
end

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

function softmax!(b::Backend,
    y::AbstractArray, x::AbstractArray;
    dims::Union{Int,Val{1}} = Val(1)
)
    if dims isa Val{1}
        x′ = reshape(x, Keep(), :)
        y′ = reshape(y, Keep(), :)
        _softmax!(b, y′, x′, dims)
        return y
    end
    return _softmax!(b, y, x, dims)
end
