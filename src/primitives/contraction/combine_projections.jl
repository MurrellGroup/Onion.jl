function _combine_projections(::DefaultBackend,
    a::AbstractArray{T,4}, b::AbstractArray{T,4}, outgoing::Bool,
) where T
    return einsum(a, b, outgoing ?
        einops"c i j b, c l j b -> c i l b" :
        einops"c j i b, c j l b -> c i l b")
end
