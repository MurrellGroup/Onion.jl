using NNlib: batched_mul, batched_transpose

function combine_projections(::DefaultBackend,
    a::AbstractArray{T,4}, b::AbstractArray{T,4}, outgoing::Bool,
) where T
    # a, b: (C, L, L, B) — channel-first pair representations
    a_perm = permutedims(a, (2, 3, 1, 4))
    b_perm = permutedims(b, (2, 3, 1, 4))
    L = size(a_perm, 1)
    C = size(a_perm, 3)
    B = size(a_perm, 4)
    a3 = reshape(a_perm, L, L, C * B)
    b3 = reshape(b_perm, L, L, C * B)
    x3 = if outgoing
        batched_mul(a3, batched_transpose(b3))
    else
        batched_mul(batched_transpose(a3), b3)
    end
    x2 = reshape(x3, L, L, C, B)
    return permutedims(x2, (3, 1, 2, 4))
end
