# Primitives are backend-dispatched kernel contracts: callable singletons (<: Function)
# that backends extend with concrete implementations.
#
#   linear!(::DefaultBackend, y, x, W, b) = ...   # mutating impl
#   linear(::DefaultBackend, x, W, b) = ...        # AD-friendly impl
#
# @primitive linear   declares a Primitive callable + dispatch chain.
# @primitive linear!  same, for mutating variant (name ending in ! detected).
#
# Dispatch chain:
#   linear(args...) → linear(Rules(), args...)
#                   → linear(backend(r, linear), r, args...)
#                   → linear(backend, args...)  ← backend impl

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

macro primitive(name)
    name_str = string(name)
    T = endswith(name_str, '!') ? Symbol('#', name_str[1:end-1], "_mut") : Symbol('#', name)

    esc(quote
        struct $T <: $Primitive end
        Base.@__doc__ const $name = $T()
        $(Expr(:public, name))
        $name(args...; kws...) =
            $name($Rules(), args...; kws...)
        $name(r::$Rules, args...; kws...) =
            $name($backend(r, $name), r, args...; kws...)
        $name(b::$Backend, r::$Rules, args...; kws...) =
            $name(b, args...; kws...)
    end)
end

include("linear.jl")
include("softmax.jl")
include("norm/norm.jl")
include("attention/attention.jl")
include("feedforward/feedforward.jl")
include("positional/rotary.jl")
include("contraction/combine_projections.jl")
include("recurrent/recurrent.jl")
