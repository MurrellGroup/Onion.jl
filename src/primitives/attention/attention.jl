using OnionStyle: Optional
using NNlib: ⊠
using Einops: rearrange, @einops_str

using ..Onion: causal_mask

apply_pad_mask(a, b::AbstractArray) = a .+ rearrange(log.(eltype(a).(b)), einops"kl ... -> kl 1 1 ...")
apply_pad_mask(a, ::Nothing) = a

apply_pair_bias(a, b::AbstractArray) = a .+ rearrange(b, einops"h ql kl ... -> kl ql h ...")
apply_pair_bias(a, ::Nothing) = a

apply_causal_mask(a, causal) = causal ? a .+ causal_mask(a) : a

@impl DefaultBackend function attention(
    q::AbstractArray{T},
    k::AbstractArray{T},
    v::AbstractArray{T};
    pair::Optional{AbstractArray{T}} = nothing,
    kpad_mask::Optional{AbstractArray} = nothing,
    causal::Bool = false,
) where T
    query_group_size = size(q, 3) ÷ size(k, 3)
    if query_group_size > 1
        k, v = repeat.((k, v), einops"d l h ... -> d l (r h) ..."; r=query_group_size)
    end
    d = size(q, 1)
    kT = rearrange(k, einops"d kl ... -> kl d ...")
    a = kT ⊠ q ./ √T(d)
    a = apply_pair_bias(a, pair)
    a = apply_pad_mask(a, kpad_mask)
    a = apply_causal_mask(a, causal)
    x = v ⊠ softmax(a)
    return x
end
