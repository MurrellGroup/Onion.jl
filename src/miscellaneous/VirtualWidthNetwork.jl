using LinearAlgebra
using ChainRulesCore
using NNlib

"""
    VirtualWidthNetwork(layer, n, m)

Wrap a sublayer (e.g. attention or FFN) with the static form of
**Generalized Hyper-Connections (GHC)**.

Given a backbone hidden size \$D\$, the over-width representation is
partitioned into `n` segments, while the backbone operates on only `m`
segments. This layer:

- **compresses** an over-width state of size \$\\frac{n}{m}D\$ down
to backbone width \$D\$ by projecting down the n segments into m segments,
- applies the wrapped layer at backbone width,
- **expands** the backbone output back to n segments,
- **carries** forward the previous over-width state with a projection
from n segments to n segments, adding it to the expanded backbone output.

See: [Virtual Width Networks](https://arxiv.org/abs/2511.11238)
"""
@concrete struct VirtualWidthNetwork <: Layer
    down; side; up
    layer
end

function VirtualWidthNetwork(layer, n, m)
    r = n - m
    down = Float32[I(m); zeros(r, m)]
    side = Float32[I(n);]
    up   = Float32[repeat(I(m), 1, fld(n, m));; I(r); zeros(r, m - r)]
    return VirtualWidthNetwork(down, side, up, layer)
end

function (vwn::VirtualWidthNetwork)(h, ::typeof(einsum))
    x  = einsum(h, vwn.down, einops"(d n) ..., n m -> (d m) ...")
    z  = vwn.layer(x)
    h′ = einsum(z, vwn.up, einops"(d m) ..., m n -> (d n) ...") +
        einsum(h, vwn.side, einops"(d n₁) ..., n₁ n₂ -> (d n₂) ...")
    return h′
end

function (vwn::VirtualWidthNetwork)(h)
    (; down, side, up, layer) = vwn
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
    vwn::VirtualWidthNetwork,
    h
)
    (; down, side, up, layer) = vwn
    n, m = size(down)
    H  = rearrange(h, einops"(d n) ... -> d n ..."; n)
    X  = batched_mul(H, down)
    x  = rearrange(X, einops"d m ... -> (d m) ...")
    z, layer_pb = rrule_via_ad(config, layer, x)
    Z  = rearrange(z, einops"(d m) ... -> d m ..."; m)
    H′ = batched_mul(Z, up)
    @mul! H′ += H * side
    h′ = rearrange(H′, einops"d n ... -> (d n) ...")

    function vwn_pullback(Δh′_raw)
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

        Δvwn = Tangent{VirtualWidthNetwork}(; down=Δdown, side=Δside, up=Δup, layer=Δlayer)
        return (Δvwn, Δh)
    end

    return h′, vwn_pullback
end
