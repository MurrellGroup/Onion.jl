#Scaled dot product attention
function sdpa(xq::AbstractArray{T}, xk::AbstractArray{T}, xv::AbstractArray{T}, mask = 0) where T
    A = softmax(batched_mul(batched_transpose(xk), xq) / sqrt(T(size(xq, 1))) .+ mask; dims=1)
    return batched_mul(xv, A)
end

#For the case where the mask differs for each element in a batch
function sdpa(xq::AbstractArray{T}, xk::AbstractArray{T}, xv::AbstractArray{T}, mask::AbstractArray{T, 3}) where T
    d1,d2 = size(xk, 2), size(xq, 2)
    A = softmax(reshape(reshape(batched_mul(batched_transpose(xk), xq) / sqrt(T(size(xq, 1))), d1, d2, :, size(mask, 3)) .+ reshape(mask, d1, d2, 1, :), d1, d2, :), dims=1)
    return batched_mul(xv, A)
end

"""
    self_att_padding_mask(padmask; T = Float32)

Takes a sequence-level `padmask` (ie. length-by-batch, where 0 indicates a padded position) and returns a (non-causal) self-attention mask
that is length-by-length-by-batch and which prevents information flow from padded positions to unpadded positions.
"""
function self_att_padding_mask(padmask; T = Float32)
    pm = T.(padmask)
    mask = log.(clamp.(reshape(pm, 1, size(pm)...) .* reshape(pm, size(pm,1), 1, size(pm,2)) .+ Diagonal(similar(padmask, size(padmask,1)) .= 1), 0, 1))
    return mask
end

"""
    cross_att_padding_mask(padmask, other_dim; T = Float32)

Takes a sequence-level `padmask` and a dimension `other_dim` and returns a cross-attention mask that is length-by-other_dim-by-batch.
This prevents information flow from padded `key` positions to any `query` positions (but ignores padding in the `query` positions, because nothing should flow out of those).
"""
function cross_att_padding_mask(padmask, other_dim; T = Float32)
    pm = T.(padmask)
    return log.(reshape(pm, size(pm,1), 1, size(pm,2)) .* (similar(pm, 1, other_dim, size(pm,2)) .= 1))
end

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
    wq
    wk
    wv
    wo
    dim::Int
    n_heads::Int
    n_kv_heads::Int
    head_dim::Int
end

Flux.@layer Attention

function Attention(dim::Int, n_heads::Int, n_kv_heads=n_heads; qkv_bias=false)
    head_dim = dim ÷ n_heads
    Attention(
        Dense(dim => n_heads * head_dim, bias=qkv_bias),
        Dense(dim => n_kv_heads * head_dim, bias=qkv_bias),
        Dense(dim => n_kv_heads * head_dim, bias=qkv_bias),
        Dense(n_heads * head_dim => dim, bias=false),
        dim,
        n_heads,
        n_kv_heads,
        head_dim
    )
end

# Backward compatibility method for self-attention with existing interface
(attn::Attention)(x::AbstractArray, start_pos, rope=identity, mask=0) =
    attn(x, x, start_pos, rope, mask)

(attn::Attention)(xq::AbstractArray, xk::AbstractArray, start_pos, rope=identity, mask=0) =
    attn(xq, xk; start_pos, rope, mask)

function (attn::Attention)(xq::AbstractArray, xk::AbstractArray=xq; start_pos=1, rope=identity, mask=false, sdpa=naive_attention)
    q = rearrange(attn.wq(xq), ((:head_dim, :heads), :len, ..) --> (:head_dim, :len, :heads, ..); attn.head_dim)
    k = rearrange(attn.wk(xk), ((:head_dim, :heads), :len, ..) --> (:head_dim, :len, :heads, ..); attn.head_dim)
    v = rearrange(attn.wv(xk), ((:head_dim, :heads), :len, ..) --> (:head_dim, :len, :heads, ..); attn.head_dim)

    # compat -- default was previously `nothing`
    isnothing(rope) && (rope = identity)
    q, k = rope(q), rope(k)

    q_per_kv = attn.n_heads ÷ attn.n_kv_heads # for multi-query attention
    k = repeat(k, (:head_dim, :len, :heads, ..) --> (:head_dim, :len, (:q_per_kv, :heads), ..); q_per_kv)
    v = repeat(v, (:head_dim, :len, :heads, ..) --> (:head_dim, :len, (:q_per_kv, :heads), ..); q_per_kv)

    output = sdpa(q, k, v; mask)
    output = rearrange(output, (:head_dim, :len, :heads, ..) --> ((:head_dim, :heads), :len, ..); heads=attn.n_heads)
    return attn.wo(output)
end

@concrete struct TransformerBlock
    attention
    feed_forward
    attention_norm
    ffn_norm
end

"""
    TransformerBlock(dim::Int, n_heads::Int, n_kv_heads::Int = n_heads, ff_hidden_dim = 4 * dim; norm_eps=1f-5, qkv_bias=false)
    TransformerBlock{Attention,FeedForward,AttentionNorm,FeedForwardNorm}

Transformer block for GQAttention (as in Llama3). No KV caching (see Jjama3.jl for KV caching).

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
function TransformerBlock(
    dim::Int, n_heads::Int, n_kv_heads::Int = n_heads, ff_hidden_dim = 4 * dim;
    norm_eps=1f-5, qkv_bias=false
)
    TransformerBlock(
        Attention(dim, n_heads, n_kv_heads; qkv_bias),
        StarGLU(dim, ff_hidden_dim),
        RMSNorm(dim, eps=norm_eps),
        RMSNorm(dim, eps=norm_eps)
    )
end

function (block::TransformerBlock)(x; kws...)
    h = x + block.attention(block.attention_norm(x), kws...)
    out = h + block.feed_forward(block.ffn_norm(h))
    return out
end

# compat
(block::TransformerBlock)(x, start_pos, rope=identity, mask=0) =
    block(x; start_pos, rope, mask)

Flux.@layer TransformerBlock


@concrete struct AdaTransformerBlock
    attention
    feed_forward
    attention_norm
    ffn_norm
end

function AdaTransformerBlock(
    dim::Int, cond_dim::Int, n_heads::Int, n_kv_heads::Int = n_heads, ff_hidden_dim = 4 * dim;
    norm_eps=1f-5, qkv_bias=false
)
    AdaTransformerBlock(
        Attention(dim, n_heads, n_kv_heads; qkv_bias),
        StarGLU(dim, ff_hidden_dim),
        AdaLN(dim, cond_dim),
        AdaLN(dim, cond_dim)
    )
end

function (block::AdaTransformerBlock)(x, cond, rope, mask)
    h = x + block.attention(block.attention_norm(x, cond); start_pos=0, rope, mask)
    out = h + block.feed_forward(block.ffn_norm(h, cond))
    return out
end

Flux.@layer AdaTransformerBlock


function causal_mask(h::AbstractArray{<:AbstractFloat})
    @ignore_derivatives begin
        _, N = size(h)
        mask = similar(h, N, N)
        fill!(mask, -Inf)
        tril!(mask, -1) # This is swapped because we're using the slightly more efficient dim setup
        return mask
    end
end

"""
    DART(transformer; mask=:causal)

"Doubly Auto-Regressive Transformer" (DART) is a convenience layer wrapping a
transformer block that can be used to model auto-regressive data represented
along two dimensions.

!!! note
    The mask acts on the flattened tokens sequence.

# Examples

```julia
julia> dart = DART(TransformerBlock(64, 8));

julia> x = randn(Float32, 64, 4, 20);

julia> dart(x) |> size
(64, 4, 20)
```
"""
@concrete struct DART
    transformer
end

function (dart::DART)(x::AbstractArray; mask=:causal, kws...)
    h = rearrange(x, (:d, :K, :L, ..) --> (:d, (:K, :L), ..))
    mask === :causal && (mask = causal_mask(h))
    return reshape(dart.transformer(h; mask, kws...), size(x))
end

Flux.@layer DART
