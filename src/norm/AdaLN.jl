"""
    AdaLN(dim::Int, cond_dim::Int)

Adaptive Layer Normalization.

```julia
aln = AdaLN(5, 3)
h = randn(Float32, 5,10,1)
cond = randn(Float32, 3,1)
h = aln(h, cond)
```
"""
@concrete struct AdaLN
    norm
    shift
    scale  
end

@layer AdaLN

AdaLN(dim::Int, cond_dim::Int) = AdaLN(Flux.LayerNorm(dim), Flux.Dense(cond_dim, dim), Flux.Dense(cond_dim, dim))

(l::AdaLN)(x, cond) = l.norm(x) .* (1 .+ glut(l.scale(cond), ndims(x), 1)) .+ glut(l.shift(cond), ndims(x), 1)
