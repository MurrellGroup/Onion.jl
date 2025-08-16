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

# transformer methods removed in favor of using closures:
# e.g. block(x; rope=x->rope(x, positions), kws...)
