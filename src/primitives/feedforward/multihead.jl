using Einops: einsum, @einops_str

function multihead_ffn(::DefaultBackend,
    Q::AbstractArray,     # (D, H, L...)
    K::AbstractArray,     # (I, D, H) — gate weight
    U::AbstractArray,     # (I, D, H) — up weight
    V::AbstractArray,     # (D, I, H) — down weight
    act,
)
    ϕ = act.(einsum(K, Q, einops"i d h, d h ... -> i h ...")) .*
             einsum(U, Q, einops"i d h, d h ... -> i h ...")
    s = einsum(V, ϕ, einops"d i h, i h ... -> d h ...")
    return s
end

using Rewrap

function multihead_ffn(::DefaultBackend,
    Q::AbstractArray,     # (D, H, L...)
    K::AbstractArray,     # (E*D_E, D, H) — gate weights, experts flattened
    U::AbstractArray,     # (E*D_E, D, H) — up weights, experts flattened
    V::AbstractArray,     # (D, E*D_E, H) — down weights, experts flattened
    act,
    R::AbstractArray,     # (E, H, L...) — router weights
)
    ϕ = act.(einsum(K, Q, einops"i d h, d h ... -> i h ...")) .*
             einsum(U, Q, einops"i d h, d h ... -> i h ...")
    # ϕ: (I, H, L...) where I = E * D_E
    # Reshape to (D_E, E, H, L...), scale by R (1, E, H, L...), flatten back
    E = size(R, 1)
    ϕ = reshape(reshape(ϕ, Split(1, (:, E)), ..) .* reshape(R, Unsqueeze(), ..), size(ϕ))
    s = einsum(V, ϕ, einops"d i h, i h ... -> d h ...")
    return s
end
