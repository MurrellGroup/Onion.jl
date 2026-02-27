"""
    ESMFoldAttention(embed_dim, num_heads, head_width; gated=false)

Fused QKV projection + `attention` primitive call.
Returns `(output, nothing)` for API compatibility with ESMFold trunk.
"""
@concrete struct ESMFoldAttention <: Layer
    proj; o_proj; g_proj
    num_heads::Int; head_width::Int; gated::Bool
end

function ESMFoldAttention(embed_dim::Int, num_heads::Int, head_width::Int; gated::Bool=false)
    proj   = Linear(embed_dim => 3 * embed_dim, bias=false)
    o_proj = Linear(embed_dim => embed_dim)
    g_proj = gated ? Linear(embed_dim => embed_dim) : nothing
    return ESMFoldAttention(proj, o_proj, g_proj, num_heads, head_width, gated)
end

function (m::ESMFoldAttention)(x; mask=nothing, bias=nothing)
    D, H = m.head_width, m.num_heads
    L, B = size(x, 2), size(x, 3)

    t = rearrange(m.proj(x), einops"(D three H) L B -> D three H L B"; D, three=3)
    # Extract Q/K/V → (D, L, H, B) for attention primitive
    q, k, v = rearrange.(@views((t[:, 1, :, :, :], t[:, 2, :, :, :], t[:, 3, :, :, :])),
                          einops"D H L B -> D L H B")

    # Build pair bias in (K, Q, H, B) format
    pair = nothing
    if bias !== nothing
        pair = rearrange(bias, einops"H Q K B -> K Q H B")
    end
    if mask !== nothing
        T = eltype(t)
        mask_bias = rearrange(ifelse.(mask .== 1, zero(T), T(-Inf)), einops"L B -> L 1 1 B")
        pair = pair === nothing ? repeat(mask_bias, 1, L, H, 1) : pair .+ mask_bias
    end

    out = attention(q, k, v; pair)
    o = rearrange(out, einops"D L H B -> (D H) L B")

    if m.gated
        o = NNlib.sigmoid.(m.g_proj(x)) .* o
    end

    return m.o_proj(o), nothing
end
