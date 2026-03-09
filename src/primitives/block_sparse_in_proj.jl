function block_sparse_in_proj(::DefaultBackend,
    w::AbstractArray{<:Any,3},    # (E, D, H)
    x::AbstractMatrix,            # (D, L)
    i::AbstractMatrix{<:Integer}, # (k, L)
)
    E, D, H = size(w)
    k, L = size(i)
    Y = similar(x, E, k * L)
    for tok in 1:L, slot in 1:k
        head = i[slot, tok]
        flat_id = (tok - 1) * k + slot
        Y[:, flat_id] .= @view(w[:, :, head]) * @view(x[:, tok])
    end
    return Y
end
