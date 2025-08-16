"""
    LayerNorm(dim::Int; eps::T=1f-6)

Layer Normalization.

```julia
ln = LayerNorm(64)
x = randn(Float32, 64, 10, 1)
y = ln(x)
```
"""
struct LayerNorm{T,A<:AbstractVector{T},B<:AbstractVector{T}}
    w::A
    b::B
    eps::T
end

Flux.@layer LayerNorm

LayerNorm(dim::Int; eps::T=1f-6) where T = LayerNorm(ones(T, dim), zeros(T, dim), eps)

(norm::LayerNorm)(x) = Ops.layer_norm(x, norm.w, norm.b; eps=norm.eps)
