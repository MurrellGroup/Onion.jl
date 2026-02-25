"""
    TriangleAttention(c_in, c_hidden, no_heads; starting=true)

Attention along one axis of a 4D pair tensor `(C, L₁, L₂, B)`.
When `starting=true`, attends along `L₁` (rows); otherwise `L₂` (columns).

Uses the existing `Attention` layer with `g1_gate=Modulator(sigmoid)` for gating.
"""
@concrete struct TriangleAttention <: Layer
    norm; bias_proj; attn; starting::Bool
end

function TriangleAttention(c_in::Int, c_hidden::Int, no_heads::Int; starting::Bool=true)
    norm      = LayerNorm(c_in)
    bias_proj = Linear(c_in => no_heads, bias=false)
    attn      = Attention(c_in, no_heads;
        head_dim = c_hidden,
        g1_gate  = Modulator(c_in => no_heads * c_hidden, sigmoid),
    )
    return TriangleAttention(norm, bias_proj, attn, starting)
end

function (m::TriangleAttention)(x; mask=nothing)
    # x: (C, L₁, L₂, B)
    # Permute so the attention axis is dim 2 and the other axis+batch are flattened
    if m.starting
        x_att = permutedims(x, (1, 3, 4, 2))  # (C, L₂, B, L₁) — attend along L₁
    else
        x_att = permutedims(x, (1, 2, 4, 3))  # (C, L₁, B, L₂) — attend along L₂
    end
    C, Ls, B_eff... = size(x_att)
    x_att = m.norm(x_att)

    # Flatten extra dims into batch: (C, L_seq, batch_eff)
    batch_eff = prod(B_eff)
    x_flat = reshape(x_att, C, Ls, batch_eff)

    # Per-key bias: (H, L_seq, batch_eff) → (K, 1, H, batch_eff) for pair format
    b = m.bias_proj(x_att)  # (H, L_seq, B_eff...)
    b = reshape(b, size(b, 1), Ls, batch_eff)
    pair = permutedims(reshape(b, size(b, 1), Ls, 1, batch_eff), (2, 3, 1, 4))  # (K, 1, H, batch_eff)

    # TODO: handle mask
    out = m.attn(x_flat; pair)

    # Unflatten and inverse permute
    out = reshape(out, C, Ls, B_eff...)
    if m.starting
        out = permutedims(out, (1, 4, 2, 3))  # (C, L₁, L₂, B)
    else
        out = permutedims(out, (1, 2, 4, 3))  # (C, L₁, L₂, B)
    end
    return out
end

TriangleAttentionStartingNode(c_in::Int, c_hidden::Int, no_heads::Int; kws...) =
    TriangleAttention(c_in, c_hidden, no_heads; starting=true, kws...)

TriangleAttentionEndingNode(c_in::Int, c_hidden::Int, no_heads::Int; kws...) =
    TriangleAttention(c_in, c_hidden, no_heads; starting=false, kws...)
