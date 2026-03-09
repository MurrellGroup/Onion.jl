using Einops: einsum, @einops_str

function scatter_proj(::DefaultBackend,
    w::AbstractArray{<:Any,3}, x::AbstractMatrix, i::AbstractMatrix,
)
    E, D = size(w, 1), size(w, 2)
    k, L = size(i)
    w_sel = reshape(w[:, :, vec(i)], E, D, k, L)
    einsum(w_sel, x, einops"e d k l, d l -> e k l")
end
