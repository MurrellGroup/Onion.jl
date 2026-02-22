"""
    x → T

Change the number type of `x` to `T`.

For `x::Number`, convert `x` to type `T`.
For `x::AbstractArray{<:Number}`, convert elements of `x` to type `T`.

"`→`" can be typed by `\\to<tab>` or `\\rightarrow<tab>`
"""
function → end
x → T::Type = T.(x) # TODO: recursively adapt/fmap. see Flux._paramtype
x::Nothing → T::Type = x

→(T::Type) = Base.Fix2(→, T)
