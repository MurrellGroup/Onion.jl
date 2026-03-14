# Primitives are backend-dispatched kernel contracts: callable singletons (<: Function)
# that backends extend with concrete implementations.
#   _linear(::DefaultBackend, x, W, b) = ...   # backend implements primitive
#   _linear(::NNopBackend, x, W, b) = ...       # another backend
#
# Interface functions are the user-facing API. They handle kwargs, reshaping,
# and argument normalization, then delegate to the primitive.
#   linear(x, W, b) → linear(backend, x, W, b) → _linear(backend, x, W, b)
#
# @primitive _kernel as interface  declares both. The interface gets a default
# pass-through to the primitive; override it to add custom logic.
# Backends only need to implement the primitive, never the interface.

abstract type Primitive <: Function end

using Base.ScopedValues: ScopedValue, with

const SCOPED_BACKEND = ScopedValue{Union{Backend, Nothing}}(nothing)
const GLOBAL_BACKEND = Ref{Union{Backend, Nothing}}(nothing)

backend() = @something(
    SCOPED_BACKEND[],
    GLOBAL_BACKEND[],
    error("no backend set")
)

resolve_backend(b::Backend, ::Primitive) = b
resolve_backend(f::Function, p::Primitive) = f(p)::Backend

backend(rules::Rules) = get(rules, :backend, backend())
backend(rules::Rules, p::Primitive) = resolve_backend(backend(rules), p)

backend!(b::Backend) = (GLOBAL_BACKEND[] = b; nothing)
withbackend(f::Function, b::Backend) = with(f, SCOPED_BACKEND => b)

macro primitive(prim, as::Symbol, wrapper)
    @assert as === :as
    T = Symbol('#', prim)
    esc(quote
        # primitive: singleton struct for backend dispatch
        struct $T <: $Primitive end
        const $prim = $T()
        $(Expr(:public, prim))
        # interface: user-facing function with backend resolution
        Base.@__doc__ function $wrapper end
        $wrapper(b::$Backend, args...; kws...) =
            $prim(b, args...; kws...)
        $wrapper(b::$Backend, r::$Rules, args...; kws...) =
            $wrapper(b, args...; kws...)
        $wrapper(r::$Rules, args...; kws...) =
            $wrapper($backend(r, $prim), r, args...; kws...)
        $wrapper(args...; kws...) =
            $wrapper($Rules(), args...; kws...)
        $(Expr(:public, wrapper))
    end)
end

"""
    linear(x::AbstractMatrix, W::AbstractMatrix, b)

Matrix multiply with optional bias: `W * x .+ b`.
`b` can be an `AbstractVector` or `false` (no bias).
"""
@primitive _linear as linear
include("linear.jl")

"""
    rms_norm(x::AbstractMatrix, w::AbstractVector; eps, offset)
    rms_norm(x::AbstractMatrix; eps)
"""
@primitive _rms_norm as rms_norm
include("norm/rms_norm.jl")

"""
    layer_norm(x::AbstractMatrix, w::AbstractVector, b::AbstractVector; eps)
"""
@primitive _layer_norm as layer_norm
include("norm/layer_norm.jl")

"""
    softmax(x::AbstractMatrix; dims=1)
"""
@primitive _softmax as softmax
include("softmax.jl")

"""
    attention(
        q, k, v;
        causal, pair,
        q_lengths, k_lengths)
"""
@primitive _attention as attention
include("attention/attention.jl")

@primitive _glu_ffn as glu_ffn
include("feedforward/glu.jl")

@primitive _multihead_ffn as multihead_ffn
include("feedforward/multihead.jl")

"""
    rotary_pos_emb(x, cos, sin)

Apply rotary positional embeddings. Splits `x` along dim 1 into halves and
applies the rotation: `[x₁·cos - x₂·sin; x₂·cos + x₁·sin]`.
"""
@primitive _rotary_pos_emb as rotary_pos_emb
include("positional/rotary.jl")

"""
    combine_projections(a, b, outgoing::Bool)

Triangle multiplication contraction. `a` and `b` are (C, L, L, B) tensors.
When `outgoing`, contracts as `a @ bᵀ` per channel×batch; otherwise `aᵀ @ b`.
"""
@primitive _combine_projections as combine_projections
include("contraction/combine_projections.jl")

"""
    newton_schulz(X, coefficients)

Quintic Newton-Schulz iteration for polar decomposition.
`coefficients` is an iterable of `(a, b, c)` tuples — one per iteration.
Each step applies `Y = aX + bXXᵀX + cXXᵀXXᵀX` (tall) or the wide variant.
"""
@primitive _newton_schulz as newton_schulz
include("newton_schulz.jl")

"""
    deltanet_recurrent(q, k, v, beta, gate, state) -> (output, state)

Gated DeltaNet recurrent step (decode). Updates state in-place:
    S = exp(gate) * S
    delta = beta * (v - Sᵀk)
    S += k ⊗ delta
    output = Sᵀq
"""
@primitive deltanet_recurrent
include("recurrent/deltanet.jl")

"""
    causal_conv1d(x, conv_state, weight, bias; silu) -> (y, conv_state)

Causal depthwise conv1d state update for decode. Shifts state, inserts x,
convolves with weight, optionally applies SiLU.
"""
@primitive causal_conv1d
include("recurrent/causal_conv1d.jl")
