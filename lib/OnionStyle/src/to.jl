using Functors: fmap

"""
    x → T

Convert the floating point precision of `x` to `T`.

For `x::Number`, convert `x` to type `T`.
For `x::AbstractArray{<:AbstractFloat}`, convert element type to `T`.
For structs (Functors), recursively convert all floating point arrays.

"`→`" can be typed by `\\to<tab>` or `\\rightarrow<tab>`
"""
function → end

_to(::Type{T}, x) where T = x
_to(::Type{T}, x::AbstractFloat) where {T<:AbstractFloat} = convert(T, x)
_to(::Type{T}, x::Complex{<:AbstractFloat}) where {T<:AbstractFloat} = convert(T, x)
_to(::Type{T}, x::AbstractArray{<:AbstractFloat}) where {T<:AbstractFloat} = convert(AbstractArray{T}, x)
_to(::Type{T}, x::AbstractArray{<:Complex{<:AbstractFloat}}) where {T<:AbstractFloat} = convert(AbstractArray{Complex{T}}, x)

x → T::Type = fmap(Base.Fix1(_to, T), x)

→(T::Type) = Base.Fix2(→, T)
