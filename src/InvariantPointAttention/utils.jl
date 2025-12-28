using StaticArrays
using Rewrap

function multireshape end

SArray_type(T::Type, size::Dims) = SArray{Tuple{size...},T,length(size),prod(size)}
SArray_type(T::Type, size::Int...) = SArray_type(T, size)

function Base.reinterpret(::typeof(multireshape), ::Type{T′}, x::AbstractArray{T}) where {Size,T,T′<:SArray{Size,T}}
    element_size = Tuple(Size.parameters)
    @assert all(i == j for (i, j) in zip(element_size, size(x)))
    reshaped_x = reshape(x, Merge(length(element_size)), ..)
    x′ = reinterpret(reshape, T′, reshaped_x)
    return x′
end

function Base.reinterpret(::typeof(multireshape), ::Type{T}, x′::AbstractArray{T′}) where {Size,T,T′<:SArray{Size,T}}
    element_size = Tuple(Size.parameters)
    reshaped_x = reinterpret(reshape, T, x′)
    x = reshape(reshaped_x, Split(1, element_size), ..)
    return x
end

as_vectors(x::AbstractArray{T}) where {T} = reinterpret(multireshape, SArray_type(T, 3), x)
as_matrices(x::AbstractArray{T}) where {T} = reinterpret(multireshape, SArray_type(T, 3, 3), x)
as_array(x::AbstractArray{<:SArray{Size,T}}) where {Size,T} = reinterpret(multireshape, T, x)

_norm(x; dims) = .√sum(abs2, x; dims)
