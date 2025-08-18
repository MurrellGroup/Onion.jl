"""
    TransformerBlock(dim::Int, n_heads::Int, n_kv_heads::Int = n_heads, ff_hidden_dim = 4 * dim; norm_eps=1f-5, qkv_bias=false)

Transformer block for GQAttention (as in Llama3).

```julia
dim = 64
n_heads = 8
n_kv_heads = 4
seqlen = 10

rope = RoPE(dim ÷ n_heads, 1000)
t = TransformerBlock(dim, n_heads, n_kv_heads)

h = randn(Float32, dim, seqlen, 1)

#Use without a mask:
h = t(h, 1, rope[1:seqlen])

#Use with a causal mask:
mask = Onion.causal_mask(h)
h = t(h, 1, rope[1:seqlen], mask)
```
"""
@concrete struct TransformerBlock
    attention
    feed_forward
    attention_norm
    ffn_norm
    pair_proj
end

Flux.@layer TransformerBlock

function TransformerBlock(
    in_dim::Int, n_heads::Int,
    n_kv_heads::Int = n_heads,
    ff_hidden_dim::Int = 4*in_dim;
    out_init_scale=1, norm_eps=1f-5,
    feed_forward = StarGLU(in_dim, ff_hidden_dim),
    attention_norm = RMSNorm(in_dim, eps=norm_eps),
    ffn_norm = RMSNorm(in_dim, eps=norm_eps),
    pair_proj = identity,
    kws...
)
    return TransformerBlock(
        Attention(in_dim, n_heads, n_kv_heads; out_init_scale, kws...),
        feed_forward, attention_norm, ffn_norm, pair_proj
    )
end

function (block::TransformerBlock)(
    x, xs...;
    cond=nothing, pair_feats=nothing,
    pair=block.pair_proj(pair_feats),
    kws...
)
    cond = isnothing(cond) ? () : (cond,)
    h = x + block.attention(
        block.attention_norm(x, cond...), xs...;
        pair, kws...)
    return h + block.feed_forward(block.ffn_norm(h, cond...))
end


function AdaTransformerBlock(
    dim::Int, cond_dim::Int, args...; kws...
)
    return TransformerBlock(
        dim, args...; kws...,
        attention_norm = AdaLN(dim, cond_dim),
        ffn_norm = AdaLN(dim, cond_dim),
    )
end


@concrete struct STRINGBlock
    block
    rope
end

Flux.@layer STRINGBlock

function STRINGBlock(block::TransformerBlock, d_coords::Int; kws...)
    rope = STRINGRoPE(block.attention.head_dim, block.attention.n_heads, d_coords; kws...)
    return STRINGBlock(block, rope)
end

function (layer::STRINGBlock)(args...; positions, kws...)
    return layer.block(args...; rope=x->layer.rope(x, positions), kws...)
end
