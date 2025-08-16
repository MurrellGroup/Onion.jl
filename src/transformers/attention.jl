"""
    Attention(dim::Int, n_heads::Int, n_kv_heads=n_heads; qkv_bias=false)

Attention layer that supports both self-attention and cross-attention (as in Llama3).

# Self-attention example
```julia
dim = 64
n_heads = 8
n_kv_heads = 4

attn = Attention(dim, n_heads, n_kv_heads)
output = attn(x)  # Self-attention
```

# Cross-attention example
```julia
output = attn(query, key, value)  # Cross-attention
```
"""
@concrete struct Attention
    wq; wk; wv; wo
    q_norm; k_norm
    in_dim::Int
    head_dim::Int
    n_heads::Int
    n_kv_heads::Int
end

Flux.@layer Attention

function Attention(
    in_dim::Int, n_heads::Int, n_kv_heads::Int=n_heads;
    head_dim = in_dim ÷ n_heads, qkv_bias=false,
    q_norm=identity, k_norm=identity,
    out_init_scale=1,
)
    wq = Dense(in_dim => n_heads * head_dim, bias=qkv_bias)
    wk = Dense(in_dim => n_kv_heads * head_dim, bias=qkv_bias)
    wv = Dense(in_dim => n_kv_heads * head_dim, bias=qkv_bias)
    wo = Dense(n_heads * head_dim => in_dim, bias=false)
    wo.weight .*= out_init_scale
    return Attention(wq, wk, wv, wo, q_norm, k_norm,
        head_dim, head_dim, n_heads, n_kv_heads)
end

function (layer::Attention)(
    xq, xk=xq;
    rope=identity, krope=rope, cache=tuple,
    sdpa=Ops.sdpa, kws...
)
    q, k, v = layer.wq(xq), layer.wk(xk), layer.wv(xk)
    q, k, v = rearrange.((q, k, v), einops"(d h) l ... -> d l h ..."; d=layer.head_dim)
    q, k = layer.q_norm(q), layer.k_norm(k)
    q, k = rope(q), krope(k)
    k, v = cache(k, v)
    k, v = repeat.((k, v), einops"d l h ... -> d l (r h) ..."; r=layer.n_heads÷layer.n_kv_heads)
    x = sdpa(q, k, v; kws...)
    x = rearrange(x, einops"d l h ... -> (d h) l ..."; h=layer.n_heads)
    return layer.wo(x)
end
