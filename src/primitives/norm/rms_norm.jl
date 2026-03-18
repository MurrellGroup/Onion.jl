using Statistics: mean

"""
    rms_norm(x::AbstractMatrix; eps)
    rms_norm(x::AbstractMatrix, w::AbstractVector; eps, offset)

RMS normalization on dim 1. Input must be 2D — callers reshape if needed.
"""
@primitive rms_norm
@primitive rms_norm!

#=== without weight ===#

lazy_rms_norm(x; eps) = @lazy x / √($mean(abs2, x; dims=1) + eps)

function rms_norm!(::DefaultBackend, y::AbstractMatrix, x::AbstractMatrix; eps)
    y .= lazy_rms_norm(x; eps)
    return y
end

get_buffers(::typeof(rms_norm), b::Backend, x; kws...) = (; y = similar(x))

function rms_norm(b::Backend, x::AbstractMatrix; eps)
    bufs = get_buffers(rms_norm, b, x; eps)
    rms_norm!(b, bufs.y, x; eps)
    return bufs.y
end

#=== with weight ===#

lazy_rms_norm(x, w; offset, kws...) = @lazy (w .+ offset) .* $lazy_rms_norm(x; kws...)

function rms_norm!(::DefaultBackend,
    y::AbstractMatrix, x::AbstractMatrix, w::AbstractVector;
    eps, offset
)
    y .= lazy_rms_norm(x, w; eps, offset)
    return y
end

get_buffers(::typeof(rms_norm), b::Backend, x, w; kws...) = (; y = similar(x))

function rms_norm(b::Backend, x::AbstractMatrix, w::AbstractVector; eps, offset)
    bufs = get_buffers(rms_norm, b, x, w; eps, offset)
    rms_norm!(b, bufs.y, x, w; eps, offset)
    return bufs.y
end
