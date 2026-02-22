using .Primitives: @primitive

"""
    rms_norm(x::AbstractMatrix, w::AbstractVector; eps, offset)
    rms_norm(x::AbstractMatrix; eps)
"""
@primitive rms_norm
include("norm/rms_norm.jl")

"""
    layer_norm(x::AbstractMatrix, w::AbstractVector, b::AbstractVector; eps)
"""
@primitive layer_norm
include("norm/layer_norm.jl")

"""
    softmax(x::AbstractMatrix)
"""
@primitive softmax
include("softmax.jl")

"""
    attention(
        q, k, v;
        causal,
        k_lengths,
        pair,
    )
"""
@primitive attention
include("attention/attention.jl")

@primitive glu_ffn
include("feedforward/glu.jl")

@primitive multihead_ffn
include("feedforward/multihead.jl")
