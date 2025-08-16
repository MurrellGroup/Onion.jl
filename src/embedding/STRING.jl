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
    As this needs to be learnable it should preferably be used with the STRINGTransformerBlock/AdaSTRINGTransformerBlock
"""
@concrete struct STRINGRoPE
    head_dim::Int
    n_heads::Int
    d_coords::Int
    thetas
    A_param
end

Flux.@layer STRINGRoPE trainable=(thetas, A_param)

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
