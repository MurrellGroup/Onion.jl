using SpecialFunctions: erf

"""
    Derf(dim; init_alpha=0.5)

Dynamic erf (Derf) layer for point-wise normalization of inputs.

See [Stronger Normalization-Free Transformers](https://arxiv.org/abs/2512.10938) for more details.
"""
@concrete struct Derf <: PointWiseNorm
    α; s; γ; β
end

function Derf(
    dim::Integer;
    init_alpha::Real = 0.5,
    T::Type{<:AbstractFloat} = Float32
)
    return Derf(
        fill(T(init_alpha)),
        zeros(T),
        ones(T, dim),
        zeros(T, dim)
    )
end

lazy_apply((; α, s, γ, β)::Derf, x) = @lazy γ * erf(α * x + s) + β
LayerStyle(::Type{<:Derf}) = LazyStyle()

@views function Base.getindex((; α, s, γ, β)::Derf, i::AbstractVector)
    Derf(α, s, γ[i], β[i])
end

Base.show(io::IO, (; γ)::Derf) = print(io, "Derf($(length(γ)))")
