# Multidimensional RoPE (STRING) from Schneck et al. 2025:
# "Learning the RoPEs: Better 2D and 3D Position Encodings with STRING".
# Link to paper: https://arxiv.org/abs/2502.02562

@info "Using local STRING.jl (dev env)"

@concrete struct STRING
    dim
    d_coords
    thetas
    orthogonal_parameter
end

Flux.@layer STRING

# dim is the dimensionality of the model (head), d_coords is the dimensionality of the position vector (often R^3)
function STRING(dim::Int, d_coords::Int)
    @assert iseven(dim) "Dimensionality (dim) must be even for STRING, dim=$dim was given."

    return STRING(
        dim,                                # Dimensionality of head
        d_coords,                           # Dimensionality of token position space
        randn(Float32, dim ÷ 2),           # Thetas
        randn(Float32, dim, dim),          # Orthogonal parameter
    )
end

# Batch‑aware variant  –  accepts x with shape (seq_len, batch)
# Returns rotation matrices of shape (2, 2, k, seq_len, batch)
function ContinuousRoPE(x::AbstractArray, rope::STRING)
    # Phase: (k, S, B)  ←  broadcast multiply
    phase = reshape(rope.thetas, :, 1, 1) .* reshape(x, 1, size(x,1), size(x,2))

    c = rearrange(cos.(phase), (..) --> (1, 1, ..))                             # (k,S,B)
    s = rearrange(sin.(phase), (..) --> (1, 1, ..))

    # Assemble rotation blocks (2,2,k,S,B) without loops
    rot = [c -s
           s  c]
    return rot
end

# Expects `position` with shape (d_coords, seq_len, batch)
# Returns a tensor with shape (dim, dim, seq_len, batch)
function (rope::STRING)(position::AbstractArray)
    @assert ndims(position) == 3 "Position must have shape (d_coords, seq_len, batch)."
    @assert size(position,1) == rope.d_coords "Coordinate dimension mismatch."

    # Sum over spatial coordinates → (seq_len, batch)
    coordinate_sum = dropdims(sum(position; dims=1), dims=1)

    # Rotation blocks: (2, 2, k, seq_len, batch)
    MultiRope = ContinuousRoPE(coordinate_sum, rope)

    # Orthogonal matrix from eq. 9
    A  = rope.orthogonal_parameter
    P  = exp(A - A')
    PT = P'                                          # (dim, dim)

    k   = size(MultiRope,3)                          # number of 2×2 blocks
    S   = size(MultiRope,4)                          # seq_len
    B   = size(MultiRope,5)                          # batch
    γ   = size(PT,2)                                 # dim

    # Align β‑index (second dim) for contraction
    D = reshape(MultiRope, 2, 2, k, S, B, 1)         # α β k S B 1
    Q = reshape(PT,        1, 2, k, 1, 1, γ)         # 1 β k 1 1 γ

    # Contract over β without explicit loops
    out = dropdims(sum(D .* Q; dims=2), dims=2)      # (2, k, S, B, γ)

    # Merge α & k → dim
    out = reshape(out, rope.dim, S, B, γ)            # (dim, S, B, γ)

    # This section should be removable, see remark under eq. 9 in STRING paper 
    # (this would however break translational invariance tests)
    
    # Left‑multiply by P for each (S,B) slice
    out2d = reshape(out, rope.dim, :)                # flatten trailing dims
    final = P * out2d                                # (dim, S*B*γ)
    final = reshape(final, rope.dim, S, B, γ)        # (dim, S, B, dim)
    final = permutedims(final, (1, 4, 2, 3))         # (dim, dim, S, B)

    return final
end

struct MultiHeadSTRING
    head_dim::Int
    n_heads::Int
    string_heads::Vector{STRING}
    premade_indexvecs::Vector{Int}
end

Flux.@layer MultiHeadSTRING trainable=(string_heads)

function MultiHeadSTRING(head_dim::Int, n_heads::Int, d_coords::Int)
    return MultiHeadSTRING(
        head_dim,
        n_heads,
        [STRING(head_dim, d_coords) for _ in 1:n_heads],
        [i for i in 1:n_heads]
    )    
end

function (layer::MultiHeadSTRING)(position)
    # position shape: (d_coords, seq_len, batch)
    # q/k shape: (head_dim, seq_len, heads, batch)

    out_heads = [head(position) for head in layer.string_heads]
    out = cat(out_heads...; dims=5)  # (dim, dim, seq_len, batch, heads)
    out = rearrange(out, (:rows, :cols, :seq_len, :batch, :heads) --> (:rows, :cols, :seq_len, :heads, :batch))
    return out
end

@concrete struct STRINGTransformerBlock
    attention
    feed_forward
    attention_norm
    ffn_norm
    rope
end

function STRINGTransformerBlock(
    dim::Int, n_heads::Int, d_coords::Int, n_kv_heads::Int = n_heads, ff_hidden_dim = 4 * dim;
    norm_eps=1f-5, qkv_bias=false, 
)

    head_dim = Int(dim / n_heads)
    STRINGTransformerBlock(
        Attention(dim, n_heads, n_kv_heads; qkv_bias),
        StarGLU(dim, ff_hidden_dim),
        RMSNorm(dim, eps=norm_eps),
        RMSNorm(dim, eps=norm_eps),
        MultiHeadSTRING(head_dim, n_heads, d_coords)
    )
end

function (block::STRINGTransformerBlock)(x; start_pos=1, mask=0, positions=nothing)
    h = x + block.attention(block.attention_norm(x), start_pos, block.rope, mask; positions=positions)
    out = h + block.feed_forward(block.ffn_norm(h))
    return out
end

# compat
(block::STRINGTransformerBlock)(x, start_pos, rope=identity, mask=0) =
    block(x; start_pos, rope, mask)

Flux.@layer STRINGTransformerBlock
