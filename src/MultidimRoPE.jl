struct MultidimRoPE
    theta::Float32
end

MultidimRoPE(; theta=10000f0) = MultidimRoPE(theta)

Flux.@layer MultidimRoPE trainable=()

"""
    MultidimRoPE(; theta=10000f0)

Multi-dimensional Rotary Position Embedding (RoPE) for 2D, 3D, or higher-dimensional
coordinate inputs. This is a fixed (non-learnable) generalization of the original
RoPE from Su et al. (2021), where each rotary pair of channels is assigned to a
specific coordinate dimension and rotated accordingly.

# Example
```julia
dim, n_heads, n_kv_heads, seqlen = 64, 8, 4, 16
t = TransformerBlock(dim, n_heads, n_kv_heads)
h = randn(Float32, dim, seqlen, 1)
mask = 0

positions = randn(Float32, 3, seqlen, 1)
rope = MultidimRoPE(theta=10000f0)

h_out = t(h, positions, rope, mask)  # self-attention with multi-dim RoPE
```
"""
function (rope::MultidimRoPE)(x::AbstractArray, positions::AbstractArray)
    x_4d = glut(x, 4, 3)
    pos_3d = glut(positions, 3, 2)
    D, S, H, B = size(x_4d)
    d_coords, S_pos, B_pos = size(pos_3d)
    @assert S == S_pos && B == B_pos "Sequence length or batch size mismatch between x and positions"
    num_pairs = D ÷ 2
    freqs = 1.0f0 ./ (rope.theta .^ (as_dense_on_device(0:2:D-1, x_4d, Float32)[1:num_pairs] ./ D))
    pos_indices = Int.((0:num_pairs-1) .% d_coords .+ 1)
    selected_pos = pos_3d[pos_indices, :, :]
    angles = reshape(freqs, num_pairs, 1, 1) .* selected_pos
    cos_vals = cos.(angles)
    sin_vals = sin.(angles)
    cos_vals = reshape(cos_vals, num_pairs, S, 1, B)
    sin_vals = reshape(sin_vals, num_pairs, S, 1, B)
    x1 = x_4d[1:D÷2, :, :, :]
    x2 = x_4d[D÷2+1:end, :, :, :]
    rotated_x = vcat(
        x1 .* cos_vals .- x2 .* sin_vals,
        x2 .* cos_vals .+ x1 .* sin_vals
    )
    return reshape(rotated_x, size(x))
end

function (block::TransformerBlock)(x::AbstractArray, positions::AbstractArray; rope::MultidimRoPE, mask=0)
    h = x + block.attention(block.attention_norm(x), positions, rope; mask)
    out = h + block.feed_forward(block.ffn_norm(h))
    return out
end
(block::TransformerBlock)(x::AbstractArray, positions::AbstractArray, rope::MultidimRoPE, mask=0) = block(x, positions; rope, mask)

function (block::AdaTransformerBlock)(x::AbstractArray, positions::AbstractArray, cond; rope::MultidimRoPE, mask=0)
    h = x + block.attention(block.attention_norm(x, cond), positions, rope; mask)
    out = h + block.feed_forward(block.ffn_norm(h, cond))
    return out
end
(block::AdaTransformerBlock)(x::AbstractArray, positions::AbstractArray, cond, rope::MultidimRoPE, mask=0) = block(x, positions, cond; rope, mask)

function (attn::Attention)(x::AbstractArray, positions::AbstractArray, rope::MultidimRoPE; mask=0)
    return attn(x, x, positions, rope; mask)
end
function (attn::Attention)(xq::AbstractArray, xk::AbstractArray, positions::AbstractArray, rope::MultidimRoPE; mask=0)
    q = rearrange(attn.wq(xq), ((:head_dim, :heads), :len, ..) --> (:head_dim, :len, :heads, ..); attn.head_dim)
    k = rearrange(attn.wk(xk), ((:head_dim, :heads), :len, ..) --> (:head_dim, :len, :heads, ..); attn.head_dim)
    v = rearrange(attn.wv(xk), ((:head_dim, :heads), :len, ..) --> (:head_dim, :len, :heads, ..); attn.head_dim)
    q, k = rope(q, positions), rope(k, positions)
    q_per_kv = attn.n_heads ÷ attn.n_kv_heads
    q_heads = rearrange(q, (:head_dim, :len, ..) --> (:head_dim, :len, (..,)))
    k_heads = repeat(k, (:head_dim, :len, ..) --> (:head_dim, :len, (:q_per_kv, ..)); q_per_kv)
    v_heads = repeat(v, (:head_dim, :len, ..) --> (:head_dim, :len, (:q_per_kv, ..)); q_per_kv)
    output = sdpa(q_heads, k_heads, v_heads, mask)
    output = rearrange(output, (:head_dim, :len, (:heads, :batch)) --> ((:head_dim, :heads), :len, :batch); heads=attn.n_heads)
    return attn.wo(output)
end