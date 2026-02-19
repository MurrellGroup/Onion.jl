module Primitives

using OnionCore: Backend, Rules

abstract type Primitive <: Function end
public Primitive

include("backend.jl")
public backend, backend!, withbackend

(p::Primitive)(b::Backend, args...; kws...) =
    throw(MethodError(p, (b, args...)))

(p::Primitive)(b::Backend, (@nospecialize r::Rules), args...; kws...) =
    p(b, args...; kws...)

(p::Primitive)(r::Rules, args...; kws...) =
    p(backend(r, p), r, args...; kws...)

(p::Primitive)(args...; kws...) =
    p(Rules(), args...; kws...)

# XXX: remove @impl now that backends are structs
#      to better communicate what's happening
# XXX: docstrings on @primitive and @impl expressions
# - docstring of @primitive becomes a spec
# - docstring of @impl contains details and potential deviations
# XXX: automatically Base.@constprop :aggressive?
include("interface.jl")
public @primitive, @impl

end
