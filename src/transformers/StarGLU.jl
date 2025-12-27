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
    dim::Int, ff_hidden_dim::Int, out_dim::Int=dim;
    act = Flux.swish,
    out_init_scale = 1
)
    w1 = Dense(dim => ff_hidden_dim, bias=false)
    w2 = Dense(ff_hidden_dim => out_dim, bias=false)
    w3 = Dense(dim => ff_hidden_dim, bias=false)
    w2.weight .*= out_init_scale
    return StarGLU(w1, w2, w3, act)
end

StarGLU((dim1, dim2)::Pair{Int,Int}; kws...) = StarGLU(dim1, dim2; kws...)
StarGLU((dim1, (dim2, dim3))::Pair{Int,Pair{Int,Int}}; kws...) = StarGLU(dim1, dim2, dim3; kws...)

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

#=
@concrete struct StarGLU <: Layer
    up_proj; gate_proj; down_proj; act
end

function StarGLU(;
    hidden_size::Int,
    intermediate_size::Int,
    hidden_act::Function = Flux.swish,
)
    return StarGLU(
        Dense(hidden_size => intermediate_size, bias=false),
        Dense(hidden_size => intermediate_size, bias=false),
        Dense(intermediate_size => hidden_size, bias=false),
        hidden_act)
end

# compat
function StarGLU(
    hidden_size::Int, intermediate_size::Int;
    act = Flux.swish,
    out_init_scale = 1
)
    layer = StarGLU(; hidden_size, intermediate_size, hidden_act=act)
    layer.down_proj.weight .*= out_init_scale
    return layer
end

const STARGLU_CHUNK_SIZE = 1024

function (layer::StarGLU{W,W,W})(
    x, chunk_size::Int = STARGLU_CHUNK_SIZE
) where W<:Dense{typeof(identity),<:DenseMatrix,Bool}
    chunk_size > 0 || throw(ArgumentError("Chunk size must be greater than 0."))
    weights = (;
        up=layer.up_proj.weight, gate=layer.gate_proj.weight,
        down=layer.down_proj.weight, act=layer.act)
    return Onion.Ops.star_glu(weights, x; chunk_size)
end

function (layer::StarGLU)(x, chunked::Bool = false)
    !chunked || throw(ArgumentError("For chunked StarGLU, pass the chunk size as an `Int`."))
    return layer.down_proj(layer.act.(layer.gate_proj(x)) .* layer.up_proj(x))
end
=#