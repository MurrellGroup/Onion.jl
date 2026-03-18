using Statistics: mean

"""
    rms_norm(x::AbstractMatrix; eps)
    rms_norm(x::AbstractMatrix, w::AbstractVector; eps, offset)

RMS normalization on dim 1. Input must be 2D — callers reshape if needed.
"""
@primitive rms_norm
@primitive rms_norm!

#=== without weight ===#

lazyrms_norm(x; eps) = @lazy x / √($mean(abs2, x; dims=1) + eps)

function rms_norm!(::DefaultBackend, y::AbstractMatrix, x::AbstractMatrix; eps)
    y .= lazyrms_norm(x; eps)
    return y
end

get_buffers(::typeof(rms_norm), b::Backend, x; kws...) = (; y = similar(x))
get_buffers(::typeof(rms_norm), b::Backend, x, w; kws...) = (; y = similar(x))

function rms_norm(::DefaultBackend, x::AbstractMatrix; eps)
    return Broadcast.materialize(lazyrms_norm(x; eps))
end

#=== with weight ===#

lazyrms_norm(x, w; offset, kws...) = @lazy (w .+ offset) .* $lazyrms_norm(x; kws...)

function rms_norm!(::DefaultBackend,
    y::AbstractMatrix, x::AbstractMatrix, w::AbstractVector;
    eps, offset
)
    y .= lazyrms_norm(x, w; eps, offset)
    return y
end

function rms_norm(::DefaultBackend,
    x::AbstractMatrix, w::AbstractVector;
    eps, offset
)
    return Broadcast.materialize(lazyrms_norm(x, w; eps, offset))
end
