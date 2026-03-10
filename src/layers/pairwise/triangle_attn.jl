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
        g1_gate  = Modulator(c_in => no_heads * c_hidden, NNlib.sigmoid),
    )
    return TriangleAttention(norm, bias_proj, attn, starting)
end

function (m::TriangleAttention)(x; mask=nothing)
    # x: (C, L₁, L₂, B)
    # Permute so the attention axis is dim 2 and the other axis+batch are flattened
    if m.starting
        x_att = rearrange(x, einops"C L1 L2 B -> C L2 B L1")   # attend along L₁
    else
        x_att = rearrange(x, einops"C L1 L2 B -> C L1 B L2")   # attend along L₂
    end
    C, Ls, B_eff... = size(x_att)
    x_att = m.norm(x_att)

    # Flatten extra dims into batch: (C, L_seq, batch_eff)
    x_flat = rearrange(x_att, einops"C Ls ... -> C Ls (...)")

    # Per-key bias: (H, L_seq, ...) → (K, 1, H, batch_eff) for pair format
    b_flat = rearrange(m.bias_proj(x_att), einops"H K ... -> K 1 H (...)")

    # TODO: handle mask
    out = m.attn(x_flat; pair=b_flat)

    # Unflatten and inverse permute
    out = reshape(out, C, Ls, B_eff...)
    if m.starting
        out = rearrange(out, einops"C L2 B L1 -> C L1 L2 B")
    else
        out = rearrange(out, einops"C L1 B L2 -> C L1 L2 B")
    end
    return out
end

TriangleAttentionStartingNode(c_in::Int, c_hidden::Int, no_heads::Int; kws...) =
    TriangleAttention(c_in, c_hidden, no_heads; starting=true, kws...)

TriangleAttentionEndingNode(c_in::Int, c_hidden::Int, no_heads::Int; kws...) =
    TriangleAttention(c_in, c_hidden, no_heads; starting=false, kws...)
