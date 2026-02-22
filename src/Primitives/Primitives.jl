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

# XXX: automatically Base.@constprop :aggressive?
include("interface.jl")
public @primitive

end
