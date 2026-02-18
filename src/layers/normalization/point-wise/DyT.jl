"""
    DyT(dim; init_alpha=0.5)

Dynamic Tanh (DyT) layer for point-wise normalization of inputs.

See [Transformers without Normalization](https://arxiv.org/abs/2503.10622) for more details.
"""
struct DyT{T<:AbstractFloat,V<:AbstractVector{T}} <: PointWiseNorm
    alpha::V   # α
    weight::V  # γ
    bias::V    # β
end

function DyT(
    dim::Integer;
    init_alpha::Real = 0.5,
    T::Type{<:AbstractFloat} = Float32
)
    return DyT(
        fill(T(init_alpha), 1),
        ones(T, dim),
        zeros(T, dim)
    )
end

LayerStyle(::Type{<:DyT}) = FusedStyle()
fuse((; α, weight, bias)::DyT, x) = @lazy weight * tanh(α * x) + bias

Base.show(io::IO, (; weight)::DyT) = print(io, "DyT($(length(weight)))")
