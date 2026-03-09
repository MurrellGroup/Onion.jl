using Einops: einsum, @einops_str

function gather_proj(::DefaultBackend,
    w::AbstractArray{<:Any,3}, z::AbstractArray{<:Any,3}, i::AbstractMatrix,
)
    D, E = size(w, 1), size(w, 2)
    k, L = size(i)
    w_sel = reshape(w[:, :, vec(i)], D, E, k, L)
    einsum(w_sel, z, einops"d e k l, e k l -> d l")
end
