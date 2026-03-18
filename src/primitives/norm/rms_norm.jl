using Statistics: mean

"""
    rms_norm(x::AbstractMatrix, w::AbstractVector; eps, offset)
    rms_norm(x::AbstractMatrix; eps)
"""
@primitive _rms_norm as rms_norm
@primitive _rms_norm! as rms_norm!

#=== without weight ===#

lazy_rms_norm(x; dims, eps) = @lazy x / √($mean(abs2, x; dims) + eps)

function _rms_norm!(::DefaultBackend,
    y::AbstractArray, x::AbstractArray, dims::Union{Int,Val{1}}; eps
)
    y .= lazy_rms_norm(x; dims=unval(dims), eps)
    return y
end

function _rms_norm(::DefaultBackend,
    x::AbstractArray, dims::Union{Int,Val{1}}; eps
)
    return Broadcast.materialize(lazy_rms_norm(x; dims=unval(dims), eps))
end

function rms_norm(b::Backend,
    x::AbstractArray;
    dims::Union{Int,Val{1}} = Val(1),
    eps
)
    if dims isa Val{1}
        x′ = reshape(x, Keep(), :)
        y′ = _rms_norm(b, x′, dims; eps)
        return reshape(y′, Keep(), Split(.., size(x)[2:end]))
    end
    return _rms_norm(b, x, dims; eps)
end

function rms_norm!(b::Backend,
    y::AbstractArray, x::AbstractArray;
    dims::Union{Int,Val{1}} = Val(1),
    eps
)
    if dims isa Val{1}
        x′ = reshape(x, Keep(), :)
        y′ = reshape(y, Keep(), :)
        _rms_norm!(b, y′, x′, dims; eps)
        return y
    end
    return _rms_norm!(b, y, x, dims; eps)
end

#=== with weight ===#

lazy_rms_norm(x, w; offset, kws...) = @lazy (w .+ offset) .* $lazy_rms_norm(x; kws...)

function _rms_norm!(::DefaultBackend,
    y::AbstractArray, x::AbstractArray, w::AbstractVector, dims::Union{Int,Val{1}};
    eps, offset
)
    y .= lazy_rms_norm(x, w; dims=unval(dims), eps, offset)
    return y
end

function _rms_norm(::DefaultBackend,
    x::AbstractArray, w::AbstractVector, dims::Union{Int,Val{1}};
    eps, offset
)
    return Broadcast.materialize(lazy_rms_norm(x, w; dims=unval(dims), eps, offset))
end

function rms_norm(b::Backend,
    x::AbstractArray, w::AbstractVector;
    dims::Union{Int,Val{1}} = Val(1),
    eps, offset
)
    if dims isa Val{1}
        x′ = reshape(x, Keep(), :)
        y′ = _rms_norm(b, x′, w, dims; eps, offset)
        return reshape(y′, Keep(), Split(.., size(x)[2:end]))
    end
    return _rms_norm(b, x, w, dims; eps, offset)
end

function rms_norm!(b::Backend,
    y::AbstractArray, x::AbstractArray, w::AbstractVector;
    dims::Union{Int,Val{1}} = Val(1),
    eps, offset
)
    if dims isa Val{1}
        x′ = reshape(x, Keep(), :)
        y′ = reshape(y, Keep(), :)
        _rms_norm!(b, y′, x′, w, dims; eps, offset)
        return y
    end
    return _rms_norm!(b, y, x, w, dims; eps, offset)
end
