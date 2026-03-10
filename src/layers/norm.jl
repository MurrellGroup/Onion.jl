abstract type Norm <: Layer end
abstract type StatisticalNorm <: Norm end
abstract type PointWiseNorm <: Norm end


# ──── LayerNorm ────

"""
    LayerNorm(dim::Int; eps::T=1f-6)

Layer Normalization.

```julia
ln = LayerNorm(64)
x = randn(Float32, 64, 10, 1)
y = ln(x)
```
"""
@concrete struct LayerNorm <: StatisticalNorm
    w
    b
    eps
end

LayerNorm(dim::Int; eps::T=1f-6) where T = LayerNorm(ones(T, dim), zeros(T, dim), eps)

(norm::LayerNorm)(x) = layer_norm(x, norm.w, norm.b; eps=norm.eps)

const LayerNormFirst = LayerNorm
const BGLayerNorm = LayerNorm


# ──── RMSNorm ────

"""
    RMSNorm(dim::Int; T=Float32, eps=1f-5, zero_centered=false)

Root Mean Square Layer Normalization. As used in Llama3.
"""
@concrete struct RMSNorm <: StatisticalNorm
    weight
    eps
    offset
end

function RMSNorm(dim::Int; T=Float32, eps=1f-5, zero_centered=false)
    weight = zero_centered ? zeros(T, dim) : ones(T, dim)
    offset = zero_centered ? one(T) : zero(T)
    RMSNorm(weight, T(eps), offset)
end

(norm::RMSNorm)(x) = rms_norm(x, norm.weight; norm.eps, norm.offset)

function fuse((; weight, offset, eps)::RMSNorm, x)
    @lazy (weight + offset) * x / √($mean(abs2, x; dims=1) + eps)
end

function Base.show(io::IO, norm::RMSNorm)
    print(io, "RMSNorm(", length(norm.weight),
        ", eps=", repr(norm.eps))
    norm.offset != 0 && print(io, ", zero_centered=", true)
    print(io, ")")
end


# ──── AdaLN ────

"""
    AdaLN(dim::Int, cond_dim::Int)

Adaptive Layer Normalization.

See also [`AdaAffine`](@ref).

```julia
aln = AdaLN(5, 3)
h = randn(Float32, 5,10,1)
cond = randn(Float32, 3,1)
h = aln(h, cond)
```
"""
@concrete struct AdaLN <: StatisticalNorm
    norm
    shift
    scale
end

AdaLN(dim::Int, cond_dim::Int) = AdaLN(LayerNorm(dim), Linear(cond_dim => dim), Linear(cond_dim => dim))

(l::AdaLN)(x, cond) = l.norm(x) .* (1 .+ glut(l.scale(cond), ndims(x), 1)) .+ glut(l.shift(cond), ndims(x), 1)


# ──── LpNorm ────

"""
    LpNorm(p; dims=1, eps=1f-6)
    LpNorm{p}(; dims=1, eps=1f-6)

A p-norm layer. This layer has no trainable parameters.

See also the [`L2Norm`](@ref) alias for `p=2`.
"""
@concrete struct LpNorm{p} <: StatisticalNorm
    dims
    eps
end

LpNorm{p}(; dims=1, eps=1f-7) where p = LpNorm{p}(dims, eps)
LpNorm(p::Int; kws...) = LpNorm{p}(; kws...)

abs3(x) = abs2(x) * abs(x)
abs4(x) = abs2(abs2(x))

((; dims, eps)::LpNorm{1})(x) = x ./ (sum(abs, x; dims) .+ ofeltype(eps, x))
((; dims, eps)::LpNorm{2})(x) = x ./ (.√sum(abs2, x; dims) .+ ofeltype(eps, x))
((; dims, eps)::LpNorm{3})(x) = x ./ (.∛sum(abs3, x; dims) .+ ofeltype(eps, x))
((; dims, eps)::LpNorm{4})(x) = x ./ (.∜sum(abs4, x; dims) .+ ofeltype(eps, x))
((; dims, eps)::LpNorm{p})(x) where p = x ./ (sum(a -> abs(a^p), x; dims) .^ (1//p) .+ ofeltype(eps, x))

"""
    L2Norm(; dims=1, eps=1f-6)

Alias for [`LpNorm`](@ref) with `p=2`.
"""
const L2Norm = LpNorm{2}


# ──── DyT ────

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


# ──── Derf ────

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

LayerStyle(::Type{<:Derf}) = FusedStyle()
fuse((; α, s, γ, β)::Derf, x) = @lazy γ * erf(α * x + s) + β

Base.show(io::IO, (; γ)::Derf) = print(io, "Derf($(length(γ)))")
