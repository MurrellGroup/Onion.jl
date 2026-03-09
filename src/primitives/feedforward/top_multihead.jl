using Einops: einsum, @einops_str

function top_multihead_ffn(::DefaultBackend,
    Q::AbstractArray{<:Any,3},    # (D, k, L)
    K::AbstractArray{<:Any,3},    # (I, D, H)
    U::AbstractArray{<:Any,3},    # (I, D, H)
    V::AbstractArray{<:Any,3},    # (D, I, H)
    act,
    i::AbstractMatrix{<:Integer}, # (k, L)
)
    D, k, L = size(Q)
    O = similar(Q)

    for l in 1:L, j in 1:k
        h = i[j, l]
        q = @view Q[:, j, l]
        O[:, j, l] = V[:, :, h] * (act.(@view(K[:, :, h]) * q) .* (@view(U[:, :, h]) * q))
    end

    return O
end

function top_multihead_ffn(::DefaultBackend,
    Q::AbstractArray{<:Any,3},    # (D, k, L)
    K::AbstractArray{<:Any,3},    # (I, D, H)
    U::AbstractArray{<:Any,3},    # (I, D, H)
    V::AbstractArray{<:Any,3},    # (D, I, H)
    act,
    i::AbstractMatrix{<:Integer}, # (k, L)
    R::AbstractArray,             # (E, k, L)
)
    D, k, L = size(Q)
    I = size(K, 1)
    E = size(R, 1)
    D_E = I ÷ E
    O = similar(Q)

    for l in 1:L, j in 1:k
        h = i[j, l]
        q = @view Q[:, j, l]
        a = act.(@view(K[:, :, h]) * q) .* (@view(U[:, :, h]) * q)
        for e_idx in 1:E
            rng = ((e_idx - 1) * D_E + 1):(e_idx * D_E)
            a[rng] .*= R[e_idx, j, l]
        end
        O[:, j, l] = V[:, :, h] * a
    end

    return O
end
