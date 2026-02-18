using Einops: einsum, @einops_str
using ..Onion: swish

@impl DefaultBackend function multihead_ffn(
    Q,     # x
    K⁽ᵍ⁾ᵀ, # W_gate
    K⁽ᵘ⁾ᵀ, # W_up
    V,     # W_down
    act,
)
    ϕ = act.(einsum(K⁽ᵍ⁾ᵀ, Q, einops"i dₕ h, dₕ h ... -> i h ...")) .*
             einsum(K⁽ᵘ⁾ᵀ, Q, einops"i dₕ h, dₕ h ... -> i h ...")
    s = einsum(V, ϕ, einops"dₕ i h, i h ... -> dₕ h ...")
    return s
end
