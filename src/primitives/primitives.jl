# Primitives are backend-dispatched kernel contracts: callable singletons (<: Function)
# that backends extend with concrete implementations.
#
#   _linear!(::DefaultBackend, y, x, W, b) = ...  # mutating (backends implement)
#   _linear(::DefaultBackend, x, W, b) = ...       # AD-friendly override
#
# A generic Backend fallback wraps _linear! for non-mutating use:
#   _linear(::Backend, x, W, b) = (y = similar(...); y = _linear!(b, ...); y)
#
# Interface functions are the user-facing API. They handle kwargs, reshaping,
# and argument normalization, then delegate to the primitive.
#   linear(x, W, b) → linear(backend, x, W, b) → _linear(backend, x, W, b)
#
# @primitive _kernel as interface  declares a primitive + interface pair.
# Names ending in ! automatically use MutPrimitive:
#   @primitive _linear  as linear   → Primitive
#   @primitive _linear! as linear!  → MutPrimitive

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
    prim_str = string(prim)
    T = endswith(prim_str, '!') ? Symbol('#', prim_str[1:end-1], "_mut") : Symbol('#', prim)

    esc(quote
        struct $T <: $Primitive end
        const $prim = $T()
        $(Expr(:public, prim))
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

include("linear.jl")
include("softmax.jl")
include("newton_schulz.jl")
include("norm/norm.jl")
include("attention/attention.jl")
include("feedforward/feedforward.jl")
include("positional/rotary.jl")
include("contraction/combine_projections.jl")
include("recurrent/recurrent.jl")
