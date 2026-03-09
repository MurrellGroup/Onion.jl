function compute_relative_distribution_perfect_correlation(binned_distribution_1, binned_distribution_2)
    K = size(binned_distribution_1, ndims(binned_distribution_1))
    out_shape = (size(binned_distribution_1)[1:end-1]..., 2 * K - 1)
    relative_distribution = zeros(eltype(binned_distribution_1), out_shape)

    zero = zeros(eltype(binned_distribution_1), size(binned_distribution_1)[1:end-1]..., 1)

    b1 = cat(zero, binned_distribution_1; dims=ndims(binned_distribution_1))
    b2 = cat(zero, binned_distribution_2; dims=ndims(binned_distribution_2))

    cumulative_1 = cumsum(b1; dims=ndims(b1))
    cumulative_2 = cumsum(b2; dims=ndims(b2))

    last_dim = ndims(cumulative_1)

    for i in 1:K
        idx = K - 1 + i
        slice1 = view(cumulative_1, ntuple(_ -> Colon(), last_dim-1)..., 1+i:size(cumulative_1, last_dim))
        slice2 = view(cumulative_2, ntuple(_ -> Colon(), last_dim-1)..., 1:size(cumulative_2, last_dim) - i)
        slice3 = view(cumulative_1, ntuple(_ -> Colon(), last_dim-1)..., i:size(cumulative_1, last_dim) - 1)
        slice4 = view(cumulative_2, ntuple(_ -> Colon(), last_dim-1)..., 1:size(cumulative_2, last_dim) - i)

        val = sum(max.(min.(slice1, slice2) .- max.(slice3, slice4), zero(eltype(binned_distribution_1))); dims=last_dim)
        val = dropdims(val; dims=last_dim)
        relative_distribution[ntuple(_ -> Colon(), ndims(relative_distribution)-1)..., idx] .= val
    end

    for i in 1:(K - 1)
        idx = K - 1 - i
        slice1 = view(cumulative_2, ntuple(_ -> Colon(), last_dim-1)..., 1+i:size(cumulative_2, last_dim))
        slice2 = view(cumulative_1, ntuple(_ -> Colon(), last_dim-1)..., 1:size(cumulative_1, last_dim) - i)
        slice3 = view(cumulative_2, ntuple(_ -> Colon(), last_dim-1)..., i:size(cumulative_2, last_dim) - 1)
        slice4 = view(cumulative_1, ntuple(_ -> Colon(), last_dim-1)..., 1:size(cumulative_1, last_dim) - i)

        val = sum(max.(min.(slice1, slice2) .- max.(slice3, slice4), zero(eltype(binned_distribution_1))); dims=last_dim)
        val = dropdims(val; dims=last_dim)
        relative_distribution[ntuple(_ -> Colon(), ndims(relative_distribution)-1)..., idx] .= val
    end

    return relative_distribution
end
