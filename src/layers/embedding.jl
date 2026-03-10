struct Embedding{W<:AbstractMatrix} <: Layer
    weight::W
end

Embedding((in, out)::Pair{Int,Int}; init = →(Float32) ∘ randn) = Embedding(init(out, in))

(m::Embedding)(x::Integer) = m.weight[:, x]
(m::Embedding)(x::AbstractVector) = NNlib.gather(m.weight, x)
(m::Embedding)(x::AbstractArray) = reshape(m(vec(x)), :, size(x)...)

(m::Embedding)(x::AbstractVector{Bool}) = m.weight * x  # usually OneHotVector
(m::Embedding)(x::AbstractMatrix{Bool}) = m.weight * x  # usually OneHotMatrix
(m::Embedding)(x::AbstractArray{Bool}) = reshape(m(reshape(x, size(x,1), :)), :, size(x)[2:end]...)

function Base.show(io::IO, m::Embedding)
    print(io, "Embedding(", size(m.weight, 2), " => ", size(m.weight, 1), ")")
end