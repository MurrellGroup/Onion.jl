"""
    Primitive <: Function

Abstract type for all primitives. Subtypes are singleton callable types
created with `@primitive`.
"""
abstract type Primitive <: Function end

macro primitive(name)
    T = Symbol('#', name)
    esc(quote
        struct $T <: $Primitive end
        Base.@__doc__ const $name = $T()
        $(Expr(:public, name))
    end)
end
