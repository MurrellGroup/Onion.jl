function block_sparse_out_proj(::DefaultBackend,
    w::AbstractArray{<:Any,3},    # (D, E, H)
    y::AbstractMatrix,            # (E, k*L)
    i::AbstractMatrix{<:Integer}, # (k, L)
)
    D, E, H = size(w)
    k, L = size(i)
    Z = fill!(similar(y, D, L), 0)
    for tok in 1:L, slot in 1:k
        head = i[slot, tok]
        flat_id = (tok - 1) * k + slot
        Z[:, tok] .+= @view(w[:, :, head]) * @view(y[:, flat_id])
    end
    return Z
end
