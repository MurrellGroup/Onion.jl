using .Primitives: @primitive

"""
    linear(x::AbstractMatrix, W::AbstractMatrix, b)

Matrix multiply with optional bias: `W * x .+ b`.
`b` can be an `AbstractVector` or `false` (no bias).
"""
@primitive linear
include("linear.jl")

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

"""
    rotary_pos_emb(x, cos, sin)

Apply rotary positional embeddings. Splits `x` along dim 1 into halves and
applies the rotation: `[x₁·cos - x₂·sin; x₂·cos + x₁·sin]`.
"""
@primitive rotary_pos_emb
include("positional/rotary.jl")

"""
    combine_projections(a, b, outgoing::Bool)

Triangle multiplication contraction. `a` and `b` are (C, L, L, B) tensors.
When `outgoing`, contracts as `a @ bᵀ` per channel×batch; otherwise `aᵀ @ b`.
"""
@primitive combine_projections
include("contraction/combine_projections.jl")
