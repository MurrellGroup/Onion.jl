"""
    Attention(
        in_dim::Int, n_heads::Int, n_kv_heads=n_heads;
        head_dim=in_dim÷n_heads, qkv_bias=false,
        q_norm=identity, k_norm=identity,
        out_init_scale=1,
    )

Attention layer that supports both self-attention and cross-attention (as in Llama3).

# Examples

## Self-attention

```julia
in_dim = 256
n_heads = 8
n_kv_heads = 4
head_dim = 64
attn = Attention(in_dim, n_heads, n_kv_heads; head_dim)

seq_len = 10
batch = 2
x = randn(in_dim, seq_len, batch)
output = attn(x)
```
"""
@concrete struct Attention
    wq; wk; wv; wo
    q_norm; k_norm; g1_gate
    in_dim::Int
    head_dim::Int
    n_heads::Int
    n_kv_heads::Int
end

@layer Attention

function Attention(
    in_dim::Int, n_heads::Int, n_kv_heads::Int=n_heads;
    head_dim = in_dim ÷ n_heads, qkv_bias=false,
    qk_norm=false,
    q_norm=qk_norm ? RMSNorm(head_dim) : identity,
    k_norm=qk_norm ? RMSNorm(head_dim) : identity,
    g1_gate=something,
    out_init_scale=1,
)
    @assert n_heads % n_kv_heads == 0 "n_heads must be divisible by n_kv_heads"
    wq = Dense(in_dim => n_heads * head_dim, bias=qkv_bias)
    wk = Dense(in_dim => n_kv_heads * head_dim, bias=qkv_bias)
    wv = Dense(in_dim => n_kv_heads * head_dim, bias=qkv_bias)
    wo = Dense(n_heads * head_dim => in_dim, bias=false)
    wo.weight .*= out_init_scale
    return Attention(wq, wk, wv, wo, q_norm, k_norm, g1_gate,
        head_dim, head_dim, n_heads, n_kv_heads)
end

function (layer::Attention)(
    xq, xk=xq, xv=xk;
    rope=identity, krope=rope, cache=tuple,
    kws...
)
    q, k, v = layer.wq(xq), layer.wk(xk), layer.wv(xv)
    q, k, v = rearrange.((q, k, v), einops"(d h) l ... -> d l h ..."; d=layer.head_dim)
    q, k = layer.q_norm(q), layer.k_norm(k)
    q, k = rope(q), krope(k)
    k, v = cache(k, v)
    k, v = repeat.((k, v), einops"d l h ... -> d l (r h) ..."; r=layer.n_heads÷layer.n_kv_heads)
    x = Ops.sdpa(q, k, v; kws...)
    x = rearrange(x, einops"d l h ... -> (d h) l ..."; h=layer.n_heads)
    x = layer.g1_gate(x, xq)
    return layer.wo(x)
end
