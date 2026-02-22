macro primitive(name::Symbol)
    T = Symbol(:primitive_, name)
    esc(quote
        struct $T <: $Primitive end
        Base.@__doc__ const $name = $T()
        $(Expr(:public, name))
    end)
end
