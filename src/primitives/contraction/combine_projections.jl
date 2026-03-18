"""
    combine_projections(a, b, outgoing::Bool)

Triangle multiplication contraction. `a` and `b` are (C, L, L, B) tensors.
When `outgoing`, contracts as `a @ bᵀ` per channel×batch; otherwise `aᵀ @ b`.
"""
@primitive combine_projections
@primitive combine_projections!

get_buffers(::typeof(combine_projections), b::Backend, a, args...) = (; out = similar(a))

function combine_projections(b::Backend,
    a::AbstractArray{T,4}, b_arr::AbstractArray{T,4}, outgoing::Bool,
) where T
    bufs = get_buffers(combine_projections, b, a, b_arr, outgoing)
    combine_projections!(b, bufs.out, a, b_arr, outgoing)
    return bufs.out
end

function combine_projections(::DefaultBackend,
    a::AbstractArray{T,4}, b::AbstractArray{T,4}, outgoing::Bool,
) where T
    return einsum(a, b, outgoing ?
        einops"c i j b, c l j b -> c i l b" :
        einops"c j i b, c j l b -> c i l b")
end
