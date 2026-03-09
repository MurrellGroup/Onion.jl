@concrete struct ESMMultiheadAttention <: Onion.Layer
    q_proj
    k_proj
    v_proj
    out_proj
    num_heads::Int
    head_dim::Int
    scaling::Float32
    dropout::Float32
    rot_emb
end

@layer ESMMultiheadAttention

function ESMMultiheadAttention(
    embed_dim::Int,
    num_heads::Int;
    dropout::Real = 0.0,
    bias::Bool = true,
    use_rotary_embeddings::Bool = false,
)
    head_dim = embed_dim ÷ num_heads
    @assert head_dim * num_heads == embed_dim "embed_dim must be divisible by num_heads"

    q_proj = Onion.Linear(embed_dim => embed_dim; bias=bias)
    k_proj = Onion.Linear(embed_dim => embed_dim; bias=bias)
    v_proj = Onion.Linear(embed_dim => embed_dim; bias=bias)
    out_proj = Onion.Linear(embed_dim => embed_dim; bias=bias)

    rot_emb = use_rotary_embeddings ? RotaryEmbedding(head_dim) : nothing

    return ESMMultiheadAttention(
        q_proj,
        k_proj,
        v_proj,
        out_proj,
        num_heads,
        head_dim,
        Float32(head_dim)^-0.5f0,
        Float32(dropout),
        rot_emb,
    )
end

function _reshape_heads(x::AbstractArray, head_dim::Int, num_heads::Int)
    d, t, b = head_dim, size(x, 2), size(x, 3)
    x = reshape(x, d, num_heads, t, b)
    x = permutedims(x, (1, 3, 2, 4))
    return reshape(x, d, t, b * num_heads)
end

function _unshape_heads(x::AbstractArray, head_dim::Int, num_heads::Int, batch::Int)
    d, t, _ = size(x)
    x = reshape(x, d, t, num_heads, batch)
    x = permutedims(x, (1, 3, 2, 4))
    return reshape(x, d * num_heads, t, batch)
end

function _apply_key_padding_mask!(attn_weights::AbstractArray, key_padding_mask)
    if key_padding_mask === nothing
        return attn_weights
    end
    s, b = size(key_padding_mask, 2), size(key_padding_mask, 1)
    mask = permutedims(key_padding_mask, (2, 1))
    mask = repeat(mask, 1, size(attn_weights, 3) ÷ b)
    bias = ifelse.(mask, -Inf, 0.0)
    bias = convert.(eltype(attn_weights), bias)
    attn_weights .+= reshape(bias, 1, s, :)
    return attn_weights
end

function (m::ESMMultiheadAttention)(
    query::AbstractArray,
    key::AbstractArray = query,
    value::AbstractArray = key;
    key_padding_mask = nothing,
    need_head_weights::Bool = false,
    _precomputed_attn_bias = nothing,
)
    D = m.head_dim
    H = m.num_heads
    C = D * H
    T = size(query, 2)
    B = size(query, 3)

    q = m.q_proj(query)
    k = m.k_proj(key)
    v = m.v_proj(value)

    # Fast path: 4D flash attention via dispatch hooks (OnionTile overrides for GPU)
    if !need_head_weights
        # Reshape to 4D: (C, T, B) → (D, H, T, B) → permute → (D, T, H, B)
        q4 = permutedims(reshape(q, D, H, T, B), (1, 3, 2, 4))
        k4 = permutedims(reshape(k, D, H, T, B), (1, 3, 2, 4))
        v4 = permutedims(reshape(v, D, H, T, B), (1, 3, 2, 4))

        # Rotary embedding via dispatch hook (fused kernel on GPU)
        if m.rot_emb !== nothing
            cos, sin = _update_cos_sin!(m.rot_emb, T, k4)
            q4 = rotary_pos_emb_forward(q4, cos, sin)
            k4 = rotary_pos_emb_forward(k4, cos, sin)
        end

        # Flash attention via dispatch hook — scaling folded in
        if _precomputed_attn_bias !== nothing
            out4 = flash_attention_bias_forward(q4, k4, v4, _precomputed_attn_bias; scale=m.scaling)
        elseif key_padding_mask !== nothing
            # Convert key_padding_mask (B, S) to 4D bias for flash attention
            s = size(key_padding_mask, 2)
            mask_col = permutedims(key_padding_mask, (2, 1))
            pad_bias = ifelse.(mask_col, eltype(q)(-Inf), zero(eltype(q)))
            pad_bias_4d = reshape(pad_bias, s, 1, 1, B)
            pad_bias_full = repeat(pad_bias_4d, 1, T, H, 1)
            out4 = flash_attention_bias_forward(q4, k4, v4, pad_bias_full; scale=m.scaling)
        else
            out4 = flash_attention_forward(q4, k4, v4; scale=m.scaling)
        end

        # Back to (C, T, B): (D, T, H, B) → permute → (D, H, T, B) → reshape
        attn = reshape(permutedims(out4, (1, 3, 2, 4)), C, T, B)
        attn = m.out_proj(attn)
        return attn, nothing
    end

    # Slow path: batched_mul for head weight extraction
    q = q .* m.scaling
    q = _reshape_heads(q, D, H)
    k = _reshape_heads(k, D, H)
    v = _reshape_heads(v, D, H)
    if m.rot_emb !== nothing
        q, k = (m.rot_emb)(q, k)
    end

    attn_weights = NNlib.batched_mul(permutedims(q, (2, 1, 3)), k)
    _apply_key_padding_mask!(attn_weights, key_padding_mask)

    attn_weights_float = NNlib.softmax(Float32.(attn_weights); dims=2)
    attn_probs = attn_weights_float

    attn = NNlib.batched_mul(attn_probs, permutedims(v, (2, 1, 3)))
    attn = permutedims(attn, (2, 1, 3))
    attn = _unshape_heads(attn, m.head_dim, m.num_heads, size(query, 3))

    attn = m.out_proj(attn)

    t, s = size(attn_weights_float, 1), size(attn_weights_float, 2)
    b = size(query, 3)
    attn_weights_out = reshape(attn_weights_float, t, s, m.num_heads, b)
    attn_weights_out = permutedims(attn_weights_out, (4, 3, 1, 2))

    return attn, attn_weights_out
end
