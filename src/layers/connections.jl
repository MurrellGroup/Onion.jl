abstract type AbstractConnection <: Layer end
abstract type AbstractSkipConnection <: AbstractConnection end

function operator end

function (sc::AbstractSkipConnection)(F, x)
    ⊕ = operator(sc)
    return F(x) ⊕ x
end

function (sc::AbstractSkipConnection)(F, x, args...; kws...)
    sc(x -> F(x, args...; kws...), x)
end


struct SkipConnection{Op} <: AbstractSkipConnection
    op::Op
end

operator(sc::SkipConnection) = sc.op


struct ResidualConnection <: AbstractSkipConnection end

operator(::ResidualConnection) = (+)


# ──── Generalized Hyper-Connections ────

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

function (ghc::GeneralizedHyperConnection)(layer, h::AbstractArray)
    x  = einsum(h, ghc.down, einops"(d n) ..., n m -> (d m) ...")
    z  = layer(x)
    h′ = einsum(z, ghc.up, einops"(d m) ..., m n -> (d n) ...") +
        einsum(h, ghc.side, einops"(d n₁) ..., n₁ n₂ -> (d n₂) ...")
    return h′
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
