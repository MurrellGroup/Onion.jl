using Statistics: mean

"""
    rms_norm(x::AbstractMatrix; eps)
    rms_norm(x::AbstractMatrix, w::AbstractVector; eps, offset)

RMS normalization on dim 1. Input must be 2D — callers reshape if needed.
"""
@primitive _rms_norm as rms_norm
@primitive _rms_norm! as rms_norm!

#=== without weight ===#

lazy_rms_norm(x; eps) = @lazy x / √($mean(abs2, x; dims=1) + eps)

function _rms_norm!(::DefaultBackend, y::AbstractMatrix, x::AbstractMatrix; eps)
    y .= lazy_rms_norm(x; eps)
    return y
end

function _rms_norm(::DefaultBackend, x::AbstractMatrix; eps)
    return Broadcast.materialize(lazy_rms_norm(x; eps))
end

#=== with weight ===#

lazy_rms_norm(x, w; offset, kws...) = @lazy (w .+ offset) .* $lazy_rms_norm(x; kws...)

function _rms_norm!(::DefaultBackend,
    y::AbstractMatrix, x::AbstractMatrix, w::AbstractVector;
    eps, offset
)
    y .= lazy_rms_norm(x, w; eps, offset)
    return y
end

function _rms_norm(::DefaultBackend,
    x::AbstractMatrix, w::AbstractVector;
    eps, offset
)
    return Broadcast.materialize(lazy_rms_norm(x, w; eps, offset))
end
