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

(norm::LayerNorm)(x) = Ops.layer_norm(x, norm.w, norm.b; eps=norm.eps)
