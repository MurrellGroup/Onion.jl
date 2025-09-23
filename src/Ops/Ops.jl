module Ops

using NNop: NNop
using NNlib: NNlib

using GPUArraysCore
using Einops
using Statistics: mean, var


softmax(x::AbstractArray) = NNlib.softmax(x)

function softmax(x::AnyGPUArray)
    y = NNop.online_softmax(reshape(x, size(x, 1), :))
    return reshape(y, size(x))
end


function rms_norm(x::AbstractArray, w::AbstractVector; eps, offset)
    y = (w .+ offset) .* x ./ .√(mean(abs2, x, dims=1) .+ eps)
    return y
end

function rms_norm(x::AnyGPUArray, w::AnyGPUVector; eps, offset)
    y = NNop.rms_norm(reshape(x, size(x, 1), :), w; ϵ=Float32(eps), offset=Float32(offset))
    return reshape(y, size(x))
end


function layer_norm(x::AbstractArray, w::AbstractVector, b::AbstractVector; eps)
    μ = mean(x; dims=1)
    σ² = var(x; dims=1, mean=μ, corrected=false)
    (x .- μ) ./ sqrt.(σ² .+ eps) .* w .+ b
end

function layer_norm(x::AnyGPUArray, w::AnyGPUVector, b::AnyGPUVector; eps)
    y = NNop.layer_norm(reshape(x, size(x, 1), :), w, b; ϵ=Float32(eps))
    return reshape(y, size(x))
end


using NNlib: ⊠
using ..Onion: causal_mask

const Maybe{T} = Union{T, Nothing}

apply_pair_bias(a, b::AbstractArray) = a .+ rearrange(b, einops"h ql kl ... -> kl ql h ...")
apply_pair_bias(a, ::Nothing) = a

apply_pad_mask(a, b::AbstractArray) = a .+ rearrange(log.(b), einops"kl ... -> kl 1 1 ...")
apply_pad_mask(a, ::Nothing) = a

apply_causal_mask(a, causal) = causal ? a .+ causal_mask(a) : a

function sdpa(
    q::AbstractArray{T}, k::AbstractArray{T}, v::AbstractArray{T};
    pair::Maybe{AbstractArray{T}} = nothing,
    kpad_mask::Maybe{AbstractArray} = nothing,
    causal::Bool = false,
) where T<:Number
    d = size(q, 1)
    kT = rearrange(k, einops"d kl ... -> kl d ...")
    a = kT ⊠ q ./ √T(d)
    a = apply_pair_bias(a, pair)
    a = apply_pad_mask(a, kpad_mask)
    a = apply_causal_mask(a, causal)
    return v ⊠ softmax(a)
end

function sdpa(
    q::AnyGPUArray, k::AnyGPUArray, v::AnyGPUArray;
    causal=false, pair=nothing, kws...
)
    NNop.flash_attention(q, k, v, pair; causal, kws...)
end

function sdpa(
    q::AnyGPUArray{<:Any,3}, k::AnyGPUArray{<:Any,3}, v::AnyGPUArray{<:Any,3};
    kws...
)
    q, k, v = rearrange.((q, k, v), einops"... -> ... 1")
    return rearrange(sdpa(q, k, v; kws...), einops"... 1 -> ...")
end

end
