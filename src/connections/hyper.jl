using LinearAlgebra
using ChainRulesCore
using NNlib

"""
    GeneralizedHyperConnection(n, m)
    (ghc::GeneralizedHyperConnection)(layer, h::AbstractArray)

Wrap a sublayer (e.g. attention or FFN) with the static form of
**Generalized Hyper-Connections (GHC)**.

Given a backbone hidden size \$D\$, the over-width representation is
partitioned into `n` segments, while the backbone operates on only `m`
segments. This layer:

- **compresses** an over-width state of size \$\\frac{n}{m}D\$ down
to backbone width \$D\$ by projecting down the n segments into m segments,
- applies `layer` at backbone width,
- **expands** the backbone output back to n segments,
- **carries** forward the previous over-width state with a projection
from n segments to n segments, adding it to the expanded backbone output.

See also [`VirtualWidthNetwork`](@ref) and [`With`](@ref).

# Examples

```jldoctest
julia> ghc = GeneralizedHyperConnection(3, 2); # hidden width is 1.5x the backbone width 

julia> h = randn(Float32, 12, 5); # hidden state is kept at 12

julia> layer = Linear(8 => 8); # backbone width is 8

julia> ghc(layer, h) |> size
(12, 5)

julia> ghc(layer, h) == ghc(h) do h
           layer(h)
       end
true
```

See: [Virtual Width Networks](https://arxiv.org/abs/2511.11238)
"""
@concrete struct GeneralizedHyperConnection <: AbstractConnection
    down; side; up
end

const GHC = GeneralizedHyperConnection

function GeneralizedHyperConnection(n, m)
    n >= m || throw(ArgumentError("n must be greater than or equal to m"))
    down = Float32[I(m); zeros(n - m, m)]
    side = Float32[I(n);]
    up   = Float32[repeat(I(m), 1, fld(n, m));; I(mod(n, m)); zeros(m - mod(n, m), mod(n, m))]
    return GeneralizedHyperConnection(down, side, up)
end

(ghc::GeneralizedHyperConnection)(layer) = Base.Fix1(ghc, layer)

function (ghc::GeneralizedHyperConnection)(layer, h::AbstractArray, ::typeof(einsum))
    x  = einsum(h, ghc.down, einops"(d n) ..., n m -> (d m) ...")
    z  = layer(x)
    h′ = einsum(z, ghc.up, einops"(d m) ..., m n -> (d n) ...") +
        einsum(h, ghc.side, einops"(d n₁) ..., n₁ n₂ -> (d n₂) ...")
    return h′
end

function (ghc::GeneralizedHyperConnection)(layer, h::AbstractArray)
    (; down, side, up) = ghc
    n, m = size(down)
    H  = rearrange(h, einops"(d n) ... -> d n ..."; n)
    X  = batched_mul(H, down)
    x  = rearrange(X, einops"d m ... -> (d m) ...")
    z  = layer(x)
    Z  = rearrange(z, einops"(d m) ... -> d m ..."; m)
    H′ = batched_mul(Z, up)
    @mul! H′ += H * side
    h′ = rearrange(H′, einops"d n ... -> (d n) ...")
    return h′
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ghc::GeneralizedHyperConnection,
    layer,
    h::AbstractArray
)
    (; down, side, up) = ghc
    n, m = size(down)
    H  = rearrange(h, einops"(d n) ... -> d n ..."; n)
    X  = batched_mul(H, down)
    x  = rearrange(X, einops"d m ... -> (d m) ...")
    z, layer_pb = rrule_via_ad(config, layer, x)
    Z  = rearrange(z, einops"(d m) ... -> d m ..."; m)
    H′ = batched_mul(Z, up)
    @mul! H′ += H * side
    h′ = rearrange(H′, einops"d n ... -> (d n) ...")

    function ghc_pullback(Δh′_raw)
        Δh′ = unthunk(Δh′_raw)
        ΔH′ = rearrange(Δh′, einops"(d n) ... -> d n ..."; n)

        Δup   = reduce(sum, mul((Z)ᵀ, ΔH′), einops"m n ... -> m n")
        Δside = reduce(sum, mul((H)ᵀ, ΔH′), einops"n₁ n₂ ... -> n₁ n₂")

        ΔZ = mul(ΔH′, (up)ᵀ)
        ΔH = mul(ΔH′, (side)ᵀ)

        Δz = rearrange(ΔZ, einops"d m ... -> (d m) ...")
        Δlayer, Δx = layer_pb(Δz)

        ΔX = rearrange(unthunk(Δx), einops"(d m) ... -> d m ..."; m)
        Δdown = reduce(sum, mul((H)ᵀ, ΔX), einops"n m ... -> n m")
        @mul! ΔH += ΔX * (down)ᵀ

        Δh = rearrange(ΔH, einops"d n ... -> (d n) ...")

        Δghc = Tangent{GeneralizedHyperConnection}(; down=Δdown, side=Δside, up=Δup)
        return (Δghc, Δlayer, Δh)
    end

    return h′, ghc_pullback
end

"""
    VirtualWidthNetwork(layer, n::Int, m::Int)

Wrap a layer with a [`GeneralizedHyperConnection`](@ref) of size `n` and `m`.

# Examples

```jldoctest
julia> model = VirtualWidthNetwork(Linear(8 => 8), 3, 2);

julia> x = randn(Float32, 12, 5);

julia> model(x) |> size
(12, 5)
```

See: [Virtual Width Networks](https://arxiv.org/abs/2511.11238)
"""
VirtualWidthNetwork(layer, n::Int, m::Int) = With(GHC(n, m), layer)
