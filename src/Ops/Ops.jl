module Ops

using NNop: NNop
using NNlib: NNlib

using GPUArraysCore
using Einops
using Statistics: mean, var


softmax(x::AbstractArray) = NNlib.softmax(x)
softmax(x::AnyGPUArray) = NNop.online_softmax(x)


function rms_norm(x::AbstractArray, w::AbstractVector; eps)
    y = x .* (w ./ .√(mean(abs2, x, dims=1) .+ eps))
    return y
end

function rms_norm(x::AnyGPUArray, w::AnyGPUVector; eps)
    y = NNop.rms_norm(reshape(x, size(x, 1), :), w; ϵ=Float32(eps))
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

function sdpa(
    q::AbstractArray{T}, k::AbstractArray{T}, v::AbstractArray{T};
    pair::Maybe{AbstractArray{T}} = nothing,
    kpad_mask::Maybe{AbstractArray} = nothing,
    causal::Bool = false,
) where T<:Number
    d = size(q, 1)
    kT = rearrange(k, einops"d kl ... -> kl d ...")
    a = kT ⊠ q ./ √T(d)
    isnothing(pair) || (a = a .+ rearrange(pair, einops"h ql kl ... -> kl ql h ..."))
    isnothing(kpad_mask) || (a = a .+ rearrange(log.(kpad_mask), einops"kl ... -> kl 1 1 ..."))
    causal && (a = a .+ causal_mask(a))
    return v ⊠ softmax(a)
end

function sdpa(q::AnyGPUArray, k::AnyGPUArray, v::AnyGPUArray; kws...)
    NNop.flash_attention(q, k, v; kws...)
end

end
