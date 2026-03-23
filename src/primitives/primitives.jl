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

backend(rules::Rules) = get(() -> backend(), rules, :backend)
backend(rules::Rules, p::Primitive) = resolve_backend(backend(rules), p)

function backend!(b::Backend)
    GLOBAL_BACKEND[] = b
    return nothing
end

withbackend(f::Function, b::Backend) = with(f, SCOPED_BACKEND => b)

(p::Primitive)(args...; kws...) = p(Rules(), args...; kws...)
(p::Primitive)(r::Rules, args...; kws...) = p(backend(r, p), r, args...; kws...)
(p::Primitive)(b::Backend, r::Rules, args...; kws...) = p(b, args...; kws...)
(p::Primitive)(b::Backend, args...; kws...) = throw(MethodError(p, (b, args...)))

macro primitive(name)
    T = Symbol('#', name)
    esc(quote
        struct $T <: $Primitive end
        Base.@__doc__ const $name = $T()
        $(Expr(:public, name))
    end)
end

"""
    get_buffers(primitive, backend, args...; kws...) -> NamedTuple

Allocate output buffers for a primitive. Returns a NamedTuple of pre-allocated
arrays sized for the given inputs. Use with the mutating variant:

    bufs = get_buffers(linear, backend, x, W, b)
    linear!(backend, bufs.y, x, W, b)

For CUDA graphs: call `get_buffers` once at setup, reuse buffers across replays.
"""
function get_buffers end
public get_buffers

include("linear.jl")
include("softmax.jl")
include("norm/norm.jl")
include("attention/attention.jl")
include("feedforward/feedforward.jl")
include("contraction/combine_projections.jl")
include("recurrent/recurrent.jl")
include("rotary.jl")
