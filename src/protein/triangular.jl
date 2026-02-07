using NNlib

# Dispatch function for OnionTile to override with cuTENSOR kernel
combine_projections_forward(a, b, outgoing) = _cpu_combine_projections(a, b, outgoing)

function _cpu_combine_projections(a::AbstractArray, b::AbstractArray, outgoing::Bool)
    a_perm = permutedims(a, (2, 3, 1, 4))
    b_perm = permutedims(b, (2, 3, 1, 4))
    L = size(a_perm, 1)
    C = size(a_perm, 3)
    B = size(a_perm, 4)
    a3 = reshape(a_perm, L, L, C * B)
    b3 = reshape(b_perm, L, L, C * B)
    if outgoing
        x3 = NNlib.batched_mul(a3, NNlib.batched_transpose(b3))
    else
        x3 = NNlib.batched_mul(NNlib.batched_transpose(a3), b3)
    end
    x2 = reshape(x3, L, L, C, B)
    x = permutedims(x2, (3, 1, 2, 4))
    return x
end

@concrete struct OFMultiheadAttention <: Onion.Layer
    linear_q
    linear_k
    linear_v
    linear_o
    linear_g
    no_heads::Int
    c_hidden::Int
    gating::Bool
    inf::Float32
end

@layer OFMultiheadAttention

function OFMultiheadAttention(c_q::Int, c_k::Int, c_v::Int, c_hidden::Int, no_heads::Int; gating::Bool=true, inf::Real=1e9)
    linear_q = LinearFirst(c_q, c_hidden * no_heads; bias=false)
    linear_k = LinearFirst(c_k, c_hidden * no_heads; bias=false)
    linear_v = LinearFirst(c_v, c_hidden * no_heads; bias=false)
    linear_o = LinearFirst(c_hidden * no_heads, c_q; bias=true)
    linear_g = gating ? LinearFirst(c_q, c_hidden * no_heads; bias=true) : nothing
    return OFMultiheadAttention(linear_q, linear_k, linear_v, linear_o, linear_g, no_heads, c_hidden, gating, Float32(inf))
end

function (m::OFMultiheadAttention)(q_x::AbstractArray, kv_x::AbstractArray;
        biases::AbstractVector=Any[], _attn_bias_flash=nothing)
    q = m.linear_q(q_x)
    k = m.linear_k(kv_x)
    v = m.linear_v(kv_x)

    C = m.c_hidden
    H = m.no_heads
    Q = size(q_x, 2)
    K = size(kv_x, 2)
    batch_shape = size(q_x)[3:end]
    B = prod(batch_shape)
    n_batch = length(batch_shape)

    q = reshape(q, C, H, Q, batch_shape...)
    k = reshape(k, C, H, K, batch_shape...)
    v = reshape(v, C, H, K, batch_shape...)

    n = ndims(q)
    perm = vcat(4:n, 2, 3, 1)
    q = permutedims(q, perm)
    k = permutedims(k, perm)
    v = permutedims(v, perm)

    q = q .* (1f0 / sqrt(Float32(C)))

    q_flat = reshape(q, B, H, Q, C)
    k_flat = reshape(k, B, H, K, C)
    v_flat = reshape(v, B, H, K, C)

    q3 = reshape(permutedims(q_flat, (3, 4, 1, 2)), Q, C, B * H)
    k3 = reshape(permutedims(k_flat, (3, 4, 1, 2)), K, C, B * H)
    v3 = reshape(permutedims(v_flat, (3, 4, 1, 2)), K, C, B * H)

    a = NNlib.batched_mul(q3, permutedims(k3, (2, 1, 3)))

    if !isempty(biases)
        a = reshape(a, Q, K, batch_shape..., H)
        a = permutedims(a, (3:(2 + n_batch)..., ndims(a), 1, 2))
        for bias in biases
            a = a .+ bias
        end
        n = ndims(a)
        b = n - 3
        a = permutedims(a, vcat(b + 2, b + 3, collect(1:b), b + 1))
        a = reshape(a, Q, K, :)
    end

    a = NNlib.softmax(a; dims=2)
    o = NNlib.batched_mul(a, v3)

    o = reshape(o, Q, C, batch_shape..., H)
    o = permutedims(o, (3:(2 + n_batch)..., ndims(o), 1, 2))

    b = n_batch
    perm_out = vcat(b + 3, b + 1, b + 2, collect(1:b))
    o = permutedims(o, perm_out)
    o = reshape(o, C * H, Q, batch_shape...)

    if m.gating
        g = NNlib.sigmoid.(m.linear_g(q_x))
        o = g .* o
    end

    return m.linear_o(o)
end

@concrete struct TriangleAttention <: Onion.Layer
    layer_norm
    linear
    mha
    starting::Bool
    inf::Float32
end

@layer TriangleAttention

function TriangleAttention(c_in::Int, c_hidden::Int, no_heads::Int; starting::Bool=true, inf::Real=1e9)
    layer_norm = LayerNormFirst(c_in)
    linear = LinearFirst(c_in, no_heads; bias=false)
    mha = OFMultiheadAttention(c_in, c_in, c_in, c_hidden, no_heads; gating=true, inf=inf)
    return TriangleAttention(layer_norm, linear, mha, starting, Float32(inf))
end

function (m::TriangleAttention)(x::AbstractArray; mask=nothing, chunk_size=nothing)
    if m.starting
        x_att = permutedims(x, (1, 3, 4, 2))
    else
        x_att = permutedims(x, (1, 2, 4, 3))
    end
    x_att = m.layer_norm(x_att)

    triangle_bias = m.linear(x_att)
    triangle_bias = permutedims(triangle_bias, (3, 1, 4, 2))
    triangle_bias = reshape(triangle_bias, size(triangle_bias, 1), 1, size(triangle_bias, 2), size(triangle_bias, 3), size(triangle_bias, 4))

    if mask !== nothing
        if m.starting
            mask_att = permutedims(mask, (2, 3, 1))
        else
            mask_att = permutedims(mask, (1, 3, 2))
        end
        mask_bias = m.inf .* (mask_att .- 1)
        mask_bias = permutedims(mask_bias, (2, 3, 1))
        mask_bias = reshape(mask_bias, size(mask_bias, 1), size(mask_bias, 2), 1, 1, size(mask_bias, 3))
        biases = [mask_bias, triangle_bias]
    else
        biases = [triangle_bias]
    end
    out = m.mha(x_att, x_att; biases=biases)

    if m.starting
        out = permutedims(out, (1, 4, 2, 3))
    else
        out = permutedims(out, (1, 2, 4, 3))
    end

    return out
end

@concrete struct TriangleMultiplicativeUpdate <: Onion.Layer
    linear_a_p
    linear_a_g
    linear_b_p
    linear_b_g
    linear_g
    linear_z
    layer_norm_in
    layer_norm_out
    outgoing::Bool
end

@layer TriangleMultiplicativeUpdate

function TriangleMultiplicativeUpdate(c_z::Int, c_hidden::Int; outgoing::Bool=true)
    linear_a_p = LinearFirst(c_z, c_hidden)
    linear_a_g = LinearFirst(c_z, c_hidden)
    linear_b_p = LinearFirst(c_z, c_hidden)
    linear_b_g = LinearFirst(c_z, c_hidden)
    linear_g = LinearFirst(c_z, c_z)
    linear_z = LinearFirst(c_hidden, c_z)
    layer_norm_in = LayerNormFirst(c_z)
    layer_norm_out = LayerNormFirst(c_hidden)
    return TriangleMultiplicativeUpdate(
        linear_a_p, linear_a_g, linear_b_p, linear_b_g,
        linear_g, linear_z, layer_norm_in, layer_norm_out, outgoing,
    )
end

function (m::TriangleMultiplicativeUpdate)(z::AbstractArray; mask=nothing)
    z_norm = m.layer_norm_in(z)

    a_g = m.linear_a_g(z_norm)
    a_p = m.linear_a_p(z_norm)
    b_g = m.linear_b_g(z_norm)
    b_p = m.linear_b_p(z_norm)
    if mask !== nothing
        mask_r = reshape(mask, 1, size(mask, 1), size(mask, 2), size(mask, 3))
        a = @. mask_r * NNlib.sigmoid(a_g) * a_p
        b = @. mask_r * NNlib.sigmoid(b_g) * b_p
    else
        a = @. NNlib.sigmoid(a_g) * a_p
        b = @. NNlib.sigmoid(b_g) * b_p
    end

    x = combine_projections_forward(a, b, m.outgoing)
    x = m.layer_norm_out(x)
    x = m.linear_z(x)
    g_raw = m.linear_g(z_norm)
    @. x = x * NNlib.sigmoid(g_raw)
    return x
end

struct TriangleMultiplicationOutgoing <: Onion.Layer
    inner::TriangleMultiplicativeUpdate
end

@layer TriangleMultiplicationOutgoing

function TriangleMultiplicationOutgoing(c_z::Int, c_hidden::Int)
    return TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate(c_z, c_hidden; outgoing=true))
end

(m::TriangleMultiplicationOutgoing)(z; mask=nothing) = m.inner(z; mask=mask)

struct TriangleMultiplicationIncoming <: Onion.Layer
    inner::TriangleMultiplicativeUpdate
end

@layer TriangleMultiplicationIncoming

function TriangleMultiplicationIncoming(c_z::Int, c_hidden::Int)
    return TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate(c_z, c_hidden; outgoing=false))
end

(m::TriangleMultiplicationIncoming)(z; mask=nothing) = m.inner(z; mask=mask)

@concrete struct TriangularSelfAttentionBlock <: Onion.Layer
    layernorm_1
    sequence_to_pair
    pair_to_sequence
    seq_attention
    tri_mul_out
    tri_mul_in
    tri_att_start
    tri_att_end
    mlp_seq
    mlp_pair
    drop
    row_drop
    col_drop
end

@layer TriangularSelfAttentionBlock

function TriangularSelfAttentionBlock(
    sequence_state_dim::Int,
    pairwise_state_dim::Int,
    sequence_head_width::Int,
    pairwise_head_width::Int;
    dropout::Real=0,
)
    sequence_num_heads = sequence_state_dim ÷ sequence_head_width
    pairwise_num_heads = pairwise_state_dim ÷ pairwise_head_width

    layernorm_1 = LayerNormFirst(sequence_state_dim)
    sequence_to_pair = SequenceToPair(sequence_state_dim, pairwise_state_dim ÷ 2, pairwise_state_dim)
    pair_to_sequence = PairToSequence(pairwise_state_dim, sequence_num_heads)

    seq_attention = ESMFoldAttention(sequence_state_dim, sequence_num_heads, sequence_head_width; gated=true)

    tri_mul_out = TriangleMultiplicationOutgoing(pairwise_state_dim, pairwise_state_dim)
    tri_mul_in = TriangleMultiplicationIncoming(pairwise_state_dim, pairwise_state_dim)
    tri_att_start = TriangleAttention(pairwise_state_dim, pairwise_head_width, pairwise_num_heads; starting=true, inf=1e9)
    tri_att_end = TriangleAttention(pairwise_state_dim, pairwise_head_width, pairwise_num_heads; starting=false, inf=1e9)

    mlp_seq = ResidueMLP(sequence_state_dim, 4 * sequence_state_dim; dropout=dropout)
    mlp_pair = ResidueMLP(pairwise_state_dim, 4 * pairwise_state_dim; dropout=dropout)

    drop = SharedDropout(dropout, 3)
    row_drop = SharedDropout(dropout * 2, 2)
    col_drop = SharedDropout(dropout * 2, 1)

    return TriangularSelfAttentionBlock(
        layernorm_1, sequence_to_pair, pair_to_sequence, seq_attention,
        tri_mul_out, tri_mul_in, tri_att_start, tri_att_end,
        mlp_seq, mlp_pair, drop, row_drop, col_drop,
    )
end

function (m::TriangularSelfAttentionBlock)(sequence_state, pairwise_state; mask=nothing, chunk_size=nothing, residue_index=nothing)
    bias = m.pair_to_sequence(pairwise_state)
    y = m.layernorm_1(sequence_state)
    y, _ = m.seq_attention(y; mask=mask, bias=bias)

    sequence_state = sequence_state .+ m.drop(y)
    sequence_state = m.mlp_seq(sequence_state)

    pairwise_state = pairwise_state .+ m.sequence_to_pair(sequence_state)

    tri_mask = mask === nothing ? nothing :
        (reshape(mask, size(mask, 1), 1, size(mask, 2)) .* reshape(mask, 1, size(mask, 1), size(mask, 2)))
    pairwise_state = pairwise_state .+ m.row_drop(m.tri_mul_out(pairwise_state; mask=tri_mask))
    pairwise_state = pairwise_state .+ m.col_drop(m.tri_mul_in(pairwise_state; mask=tri_mask))
    pairwise_state = pairwise_state .+ m.row_drop(m.tri_att_start(pairwise_state; mask=tri_mask))
    pairwise_state = pairwise_state .+ m.col_drop(m.tri_att_end(pairwise_state; mask=tri_mask))

    pairwise_state = m.mlp_pair(pairwise_state)

    return sequence_state, pairwise_state
end
