"""
    StarGLU(dim::Int, ff_hidden_dim::Int; act=Flux.swish)

Gated Linear Unit with flexible activation function (default: `swish`, making it a SwiGLU layer as used in Llama3).

```julia
l = StarGLU(6, 8)
h = randn(Float32, 6, 10, 1)
h = l(h)
```
"""
struct StarGLU{W, F}
    w1::W
    w2::W
    w3::W
    act::F
end

@layer StarGLU

function StarGLU(
    dim::Int, ff_hidden_dim::Int;
    act = Flux.swish,
    out_init_scale = 1
)
    w1 = Linear(dim => ff_hidden_dim, bias=false)
    w2 = Linear(ff_hidden_dim => dim, bias=false)
    w3 = Linear(dim => ff_hidden_dim, bias=false)
    w2.weight .*= out_init_scale
    return StarGLU(w1, w2, w3, act)
end

const STARGLU_CHUNK_SIZE = 1024

function (ff::StarGLU)(x, chunk_size::Int = STARGLU_CHUNK_SIZE)
    chunk_size > 0 || throw(ArgumentError("Chunk size must be greater than 0."))
    layer = (;
        up = ff.w3.weight,
        gate = ff.w1.weight,
        down = ff.w2.weight,
        act = ff.act,
    )
    return Ops.star_glu(layer, x; chunk_size)
end

function (ff::StarGLU)(x, chunked::Bool)
    !chunked || throw(ArgumentError("For chunked StarGLU, pass the chunk size as an `Int`."))
    return ff.w2(ff.act.(ff.w1(x)) .* ff.w3(x))
end
