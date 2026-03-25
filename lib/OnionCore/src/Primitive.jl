"""
    Primitive <: Function

Abstract type for all primitives. Subtypes are singleton callable types
created with `@primitive` and dispatched by [`Backend`](@ref).
"""
abstract type Primitive <: Function end

(p::Primitive)(b::Backend, args...; kws...) = throw(MethodError(p, (b, args...)))

macro primitive(name)
    T = Symbol('#', name)
    esc(quote
        struct $T <: $Primitive end
        Base.@__doc__ const $name = $T()
        $(Expr(:public, name))
    end)
end
