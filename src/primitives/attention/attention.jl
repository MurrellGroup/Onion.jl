using OnionStyle: Optional
using NNlib: ⊠
using Einops: rearrange, @einops_str

using ..Onion: causal_mask

apply_pad_mask(a, b::AbstractArray) = a .+ rearrange(log.(eltype(a).(b)), einops"kl ... -> kl 1 1 ...")
apply_pad_mask(a, ::Nothing) = a

apply_pair_bias(a, b::AbstractArray) = a .+ b#rearrange(b, einops"h ql kl ... -> kl ql h ...")
apply_pair_bias(a, ::Nothing) = a

apply_causal_mask(a, causal) = causal ? a .+ causal_mask(a) : a

k_lengths_to_mask(k_lengths, kl::Int) = (1:kl) .<= reshape(k_lengths, 1, :)

function _attention(::DefaultBackend,
    q::AbstractArray{T,4},
    k::AbstractArray{T,4},
    v::AbstractArray{T,4};
    pair::Optional{AbstractArray{T}} = nothing,
    kpad_mask::Optional{AbstractArray} = nothing,
    k_lengths::Optional{AbstractVector} = nothing,
    causal::Bool = false,
) where T
    if k_lengths !== nothing
        len_mask = k_lengths_to_mask(k_lengths, size(k, 2))
        kpad_mask = isnothing(kpad_mask) ? len_mask : kpad_mask .& len_mask
    end
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
    x = v ⊠ NNlib.softmax(a)
    return x
end

function attention(b::Backend,
    q::AbstractArray{T,4},
    k::AbstractArray{T,4},
    v::AbstractArray{T,4};
    kws...
) where T
    return _attention(b, q, k, v; kws...)
end

function attention(b::Backend,
    q::AbstractArray{T,3},
    k::AbstractArray{T,3},
    v::AbstractArray{T,3};
    kws...
) where T
    q, k, v = rearrange.((q, k, v), einops"d l h -> d l h 1")
    x = attention(b, q, k, v; kws...)
    return rearrange(x, einops"d l h 1 -> d l h")
end
