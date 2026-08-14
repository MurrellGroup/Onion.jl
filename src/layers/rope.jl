# ──── RoPE ────

"""
    RoPE(dim::Int, max_length; theta::T=10000f0)

Rotary Position Embeddings (as in Llama3).

```julia
dim = 64
n_heads = 8
n_kv_heads = 4
seqlen = 10

t = TransformerBlock(dim, n_heads, n_kv_heads)
h = randn(Float32, dim, seqlen, 1)

rope = RoPE(dim ÷ n_heads, 1000)
h = t(h, 1, rope[1:seqlen]) #Note the subsetting to match seqlen
```
"""
struct RoPE{A<:AbstractArray} <: Layer
    cos::A
    sin::A
end

trainable(::RoPE) = (;)

Base.getindex(rope::RoPE, i) = RoPE(selectdim(rope.cos, 2, i), selectdim(rope.sin, 2, i))

function apply_scaling!(freqs::AbstractVector; scale_factor=8)
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    for (i, freq) in enumerate(freqs)
        wavelen = 2π / freq
        if wavelen > low_freq_wavelen
            freqs[i] = freq / scale_factor
        elseif wavelen > high_freq_wavelen
            @assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) /
                    (high_freq_factor - low_freq_factor)
            freqs[i] = (1 - smooth) * freq / scale_factor + smooth * freq
        end
    end
    return freqs
end

function RoPE(
    dim::Int, end_pos::Int;
    T=Float32,
    theta=10000f0, use_scaled=false, scale_factor=8, start_pos=0
)
    theta = T(theta)
    freqs = 1f0 ./ (theta .^ (T.(0:2:dim-1)[1:dim÷2] ./ dim))
    use_scaled && apply_scaling!(freqs; scale_factor)
    freqs_complex = transpose(cis.(T.(start_pos:end_pos-1) * freqs'))
    return RoPE(real(freqs_complex), imag(freqs_complex))
end

# Note about Huggingface weights and rotary embeddings:
# https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509
# Use this one if you're using the Hugging Face weights.
(rope::RoPE)(x) = rotary_pos_emb(x, rope.cos, rope.sin)


# ──── MultidimRoPE ────

struct MultidimRoPE <: Layer
    theta::Float32
end

MultidimRoPE(; theta=10000f0) = MultidimRoPE(theta)

trainable(::MultidimRoPE) = (;)

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
    freqs = 1.0f0 ./ (rope.theta .^ (like(0:2:D-1, x, Float32)[1:num_pairs] ./ D))
    # Zygote traces mod1 as mod, so 1D coords gather at index 0. Axis map is discrete.
    pos_indices = @ignore_derivatives mod1.(like(1:num_pairs, x), d_coords)
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


# ──── STRINGRoPE ────

# Multidimensional RoPE (STRING) from Schneck et al. 2025:
# "Learning the RoPEs: Better 2D and 3D Position Encodings with STRING".
# Link to paper: https://arxiv.org/abs/2502.02562

"""
    STRINGRoPE(head_dim::Int, n_heads::Int, d_coords::Int; init_scale=0.001f0, theta=10000f0)

Multidimensional, learnable Rotary Position Embedding (RoPE) from Schneck et al. (2025),
"Learning the RoPEs: Better 2D and 3D Position Encodings with STRING".

# Example
```julia
head_dim = 64
n_heads = 8
d_coords = 3
rope = STRINGRoPE(head_dim, n_heads, d_coords)

x = rand(Float32, head_dim, 16, n_heads, 2)      # (head_dim, seq_len, n_heads, batch)
positions = rand(Float32, d_coords, 16, 2)       # (d_coords, seq_len, batch)
x_rot = rope(x, positions)
```
!!! note
    As this needs to be learnable it should preferably be used with the STRINGBlock.
"""
@concrete struct STRINGRoPE <: Layer
    head_dim::Int
    n_heads::Int
    d_coords::Int
    thetas
    A_param
end

function STRINGRoPE(head_dim::Int, n_heads::Int, d_coords::Int; init_scale=0.001f0, theta=10000f0)
    @assert iseven(head_dim) "Head dimension must be even."
    num_pairs = head_dim ÷ 2
    freqs = 1.0f0 ./ (theta .^ (Float32.(0:2:head_dim-1)[1:num_pairs] ./ head_dim))
    thetas = repeat(reshape(freqs, :, 1, 1), 1, d_coords, n_heads)
    A_param = randn(Float32, head_dim, head_dim, n_heads) * init_scale
    STRINGRoPE(head_dim, n_heads, d_coords, thetas, A_param)
end

function apply_string_rotation(x::AbstractArray, positions::AbstractArray, thetas::AbstractArray)
    head_dim, seq_len, n_heads, batch = size(x)
    @assert iseven(head_dim)
    num_pairs = head_dim ÷ 2
    d_coords = size(positions, 1)
    x1 = x[1:num_pairs, :, :, :]
    x2 = x[num_pairs+1:end, :, :, :]
    angles = reshape(thetas, num_pairs, d_coords, 1, n_heads, 1) .*
             reshape(positions, 1, d_coords, seq_len, 1, batch)
    cumulative_angles = sum(angles, dims=2)[:, 1, :, :, :]
    cos_vals = cos.(cumulative_angles)
    sin_vals = sin.(cumulative_angles)
    rot1 = x1 .* cos_vals .- x2 .* sin_vals
    rot2 = x2 .* cos_vals .+ x1 .* sin_vals
    return vcat(rot1, rot2)
end

function (rope::STRINGRoPE)(x::AbstractArray, positions::AbstractArray)
    S = (rope.A_param .- batched_transpose(rope.A_param)) ./ 2
    x_4d = glut(x, 4, 3)
    pos_3d = glut(positions, 3, 2)
    head_dim, seq_len, n_heads, total_batch = size(x_4d)
    x_flat = reshape(x_4d, head_dim, seq_len * total_batch, n_heads)
    S_vec = eachslice(S; dims = 3)
    X_vec = eachslice(x_flat; dims = 3)
    solved = map((Sh,X) -> (I + Sh) \ X, S_vec, X_vec)
    products = map((Sh,X̂) -> (I - Sh) * X̂, S_vec, solved)
    x_transformed = cat(products...; dims = 3)
    x_restored = reshape(x_transformed, size(x_4d))
    rope_result = apply_string_rotation(x_restored, pos_3d, rope.thetas)
    return reshape(rope_result, size(x))
end
