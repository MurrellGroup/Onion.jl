"""
    AttentionPairBias(c_s, c_z, num_heads; compute_pair_bias=true, use_qk_norm=false)

Wraps `Attention` with pair-bias projection from pairwise state `z`.

    (apb)(s, z, mask, k_in=s)

- `s`: sequence state `(C_s, L, B)`
- `z`: pairwise state `(C_z, L, L, B)` (or pre-computed bias if `compute_pair_bias=false`)
- `mask`: key padding mask `(L, B)`, 1 = valid
- `k_in`: cross-attention key source (defaults to `s`)
"""
@concrete struct AttentionPairBias <: Layer
    attn
    proj_z_norm
    proj_z
    compute_pair_bias::Bool
end

function AttentionPairBias(
    c_s::Int, c_z::Int, num_heads::Int;
    compute_pair_bias::Bool=true,
    use_qk_norm::Bool=false,
)
    attn = Attention(c_s, num_heads;
        qkv_bias = true,
        qk_norm  = use_qk_norm,
        g1_gate  = Modulator(c_s => num_heads * (c_s ÷ num_heads), NNlib.sigmoid),
    )
    if compute_pair_bias
        proj_z_norm = LayerNorm(c_z)
        proj_z      = Linear(c_z => num_heads, bias=false)
    else
        proj_z_norm = identity
        proj_z      = identity
    end
    return AttentionPairBias(attn, proj_z_norm, proj_z, compute_pair_bias)
end

function (l::AttentionPairBias)(s, z, mask, k_in=s)
    # Compute pair bias: (C_z, L, L, B) → (H, L, L, B) → (K, Q, H, B)
    if l.compute_pair_bias
        pair = l.proj_z(l.proj_z_norm(z))           # (H, K, Q, B)
    else
        pair = z                                     # assume z already (H, K, Q, B)
    end
    pair = rearrange(pair, einops"H K Q B -> K Q H B")

    return l.attn(s, k_in; pair, kpad_mask=mask)
end
