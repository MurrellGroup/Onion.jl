using Einops: einsum, @einops_str

@primitive multihead_ffn
@primitive multihead_ffn!

function multihead_ffn!(::DefaultBackend, out::AbstractArray, args...)
    result = multihead_ffn(DefaultBackend(), args...)
    copyto!(out, result)
    return out
end

get_buffers(::typeof(multihead_ffn), b::Backend, Q, args...) = (; out = similar(Q))

function multihead_ffn(b::Backend, Q::AbstractArray, args...)
    bufs = get_buffers(multihead_ffn, b, Q, args...)
    multihead_ffn!(b, bufs.out, Q, args...)
    return bufs.out
end

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
    E = size(R, 1)
    ϕ = reshape(reshape(ϕ, Split(1, (:, E)), ..) .* reshape(R, Unsqueeze(), ..), size(ϕ))
    s = einsum(V, ϕ, einops"d i h, i h ... -> d h ...")
    return s
end
