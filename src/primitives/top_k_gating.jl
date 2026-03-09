function top_k_gating(::DefaultBackend, scores::AbstractMatrix, k::Integer)
    T = eltype(scores)
    H, L = size(scores)

    s = copy(scores)
    indices = similar(scores, Int32, k, L)
    values = similar(scores, k, L)

    # (H, 1) row index vector — Int32 for index extraction
    h_idx = similar(scores, Int32, H, 1)
    h_idx .= Int32.(reshape(1:H, H, 1))

    for j in 1:k
        vals = maximum(s; dims=1)                       # (1, L)
        is_max = s .== vals                              # (H, L)
        first_max = is_max .& (cumsum(is_max; dims=1) .== 1)
        idx = sum(h_idx .* first_max; dims=1)           # (1, L) Int32

        indices[j:j, :] .= idx
        values[j:j, :] .= vals
        s .= ifelse.(first_max, T(-Inf), s)
    end

    return indices, values
end
