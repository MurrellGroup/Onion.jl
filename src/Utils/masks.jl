"""
    self_att_padding_mask(padmask; T=Float32)

Takes a sequence-level `padmask` (ie. length-by-batch, where 0 indicates a padded position) and returns a (non-causal) self-attention mask
that is length-by-length-by-batch and which prevents information flow from padded positions to unpadded positions.

# Examples
```jldoctest
julia> self_att_padding_mask([1 1; 1 1; 1 0])
3×3×2 Array{Float32, 3}:
[:, :, 1] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 2] =
   0.0    0.0  -Inf
   0.0    0.0  -Inf
 -Inf   -Inf     0.0
```
"""
function self_att_padding_mask(padmask; T=Float32)
    pm = T.(padmask)
    return log.(clamp.(
        rearrange(pm, einops"n ... -> 1 n ...") .*
        rearrange(pm, einops"n ... -> n 1 ...") .+
        Diagonal(ones_like(pm, size(pm, 1))),
        0, 1))
end

function pad_lengths(lengths::AbstractArray{Int}, max_length=maximum(lengths))
    padmask = zeros_like(lengths, Bool, max_length, size(lengths)...)
    for (l, i) in zip(lengths, CartesianIndices(lengths))
        padmask[1:l, i] .= true
    end
    return padmask
end

"""
    cross_att_padding_mask(padmask, other_dim; T=Float32)

Takes a sequence-level `padmask` and a dimension `other_dim` and returns a cross-attention mask that is length-by-other_dim-by-batch.
This prevents information flow from padded `key` positions to any `query` positions (but ignores padding in the `query` positions, because nothing should flow out of those).

# Examples
```jldoctest
julia> cross_att_padding_mask([1 1; 1 1; 1 0], 4)
3×4×2 Array{Float32, 3}:
[:, :, 1] =
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

[:, :, 2] =
   0.0    0.0    0.0    0.0
   0.0    0.0    0.0    0.0
 -Inf   -Inf   -Inf   -Inf
```
"""
function cross_att_padding_mask(padmask, other_dim; T=Float32)
    pm = T.(padmask)
    return log.(repeat(pm, einops"n ... -> n m ..."; m=other_dim))
end

function causal_mask(x::AbstractArray{<:AbstractFloat})
    n = size(x, 2)
    mask = like(-Inf, x, n, n)
    tril!(mask, -1) # This is swapped because we're using the slightly more efficient dim setup
    return mask
end

@non_differentiable pad_lengths(::Any...)
@non_differentiable self_att_padding_mask(::Any...)
@non_differentiable cross_att_padding_mask(::Any...)
@non_differentiable causal_mask(::Any...)
