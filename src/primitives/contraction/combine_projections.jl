"""
    combine_projections(a, b, outgoing::Bool)

Triangle multiplication contraction. `a` and `b` are (C, L, L, B) tensors.
When `outgoing`, contracts as `a @ bᵀ` per channel×batch; otherwise `aᵀ @ b`.
"""
@primitive combine_projections
@primitive combine_projections!

function combine_projections!(::DefaultBackend,
    out::AbstractArray{T,4}, a::AbstractArray{T,4}, b::AbstractArray{T,4}, outgoing::Bool,
) where T
    result = combine_projections(DefaultBackend(), a, b, outgoing)
    copyto!(out, result)
    return out
end

function combine_projections(::DefaultBackend,
    a::AbstractArray{T,4}, b::AbstractArray{T,4}, outgoing::Bool,
) where T
    return einsum(a, b, outgoing ?
        einops"c i j b, c l j b -> c i l b" :
        einops"c j i b, c j l b -> c i l b")
end
