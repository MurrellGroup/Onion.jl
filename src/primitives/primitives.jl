abstract type Primitive <: Function end

using Base.ScopedValues: ScopedValue, with

const CURRENT_BACKEND = ScopedValue{Union{Backend, Nothing}}(nothing)
const GLOBAL_BACKEND = Ref{Union{Backend, Nothing}}(nothing)

backend() = @something(
    CURRENT_BACKEND[],
    GLOBAL_BACKEND[],
    error("no backend set")
)

resolve_backend(b::Backend, ::Primitive) = b
resolve_backend(f::Function, p::Primitive) = f(p)::Backend

backend(rules::Rules) = get(rules, :backend, backend())
backend(rules::Rules, p::Primitive) = resolve_backend(backend(rules), p)

backend!(b::Backend) = (GLOBAL_BACKEND[] = b; nothing)
withbackend(f::Function, b::Backend) = with(f, CURRENT_BACKEND => b)

(p::Primitive)(b::Backend, args...; kws...) =
    throw(MethodError(p, (b, args...)))

(p::Primitive)(b::Backend, (@nospecialize r::Rules), args...; kws...) =
    p(b, args...; kws...)

(p::Primitive)(r::Rules, args...; kws...) =
    p(backend(r, p), r, args...; kws...)

(p::Primitive)(args...; kws...) =
    p(Rules(), args...; kws...)

macro primitive(name::Symbol)
    T = Symbol(:primitive_, name)
    esc(quote
        struct $T <: $Primitive end
        Base.@__doc__ const $name = $T()
        $(Expr(:public, name))
    end)
end

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
        causal, pair,
        q_lengths, k_lengths)
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

"""
    newton_schulz(X, coefficients)

Quintic Newton-Schulz iteration for polar decomposition.
`coefficients` is an iterable of `(a, b, c)` tuples — one per iteration.
Each step applies `Y = aX + bXXᵀX + cXXᵀXXᵀX` (tall) or the wide variant.
"""
@primitive newton_schulz
include("newton_schulz.jl")
