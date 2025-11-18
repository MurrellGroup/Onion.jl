module Ops

using NNop: NNop
using NNlib: NNlib

using GPUArraysCore
using Einops
using Statistics: mean, var
using LinearAlgebra


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

include("StarGLU.jl")

using NNlib: ⊠
using ..Onion.Utils: causal_mask

const Maybe{T} = Union{T, Nothing}

apply_pair_bias(a, b::AbstractArray) = a .+ rearrange(b, einops"h ql kl ... -> kl ql h ...")
apply_pair_bias(a, ::Nothing) = a

apply_pad_mask(a, b::AbstractArray) = a .+ rearrange(log.(eltype(a).(b)), einops"kl ... -> kl 1 1 ...")
apply_pad_mask(a, ::Nothing) = a

apply_causal_mask(a, causal) = causal ? a .+ causal_mask(a) : a

function naive_attention(
    q::AbstractArray{T}, k::AbstractArray{T}, v::AbstractArray{T};
    pair::Maybe{AbstractArray{T}} = nothing,
    kpad_mask::Maybe{AbstractArray} = nothing,
    causal::Bool = false,
) where T<:Number
    q, k, v = rearrange.((q, k, v), einops"d l h ... -> d l h (...)")
    k, v = repeat.((k, v), einops"d l h ... -> d l (r h) ..."; r=size(q, 3) ÷ size(k, 3))
    d = size(q, 1)
    kT = rearrange(k, einops"d kl ... -> kl d ...")
    a = kT ⊠ q ./ √T(d)
    a = apply_pair_bias(a, pair)
    a = apply_pad_mask(a, kpad_mask)
    a = apply_causal_mask(a, causal)
    x = v ⊠ softmax(a)
    return reshape(x, size(q))
end

function flash_attention(
    q::AnyGPUArray, k::AnyGPUArray, v::AnyGPUArray;
    causal=false, pair=nothing, kws...
)
    q, k, v = rearrange.((q, k, v), einops"d l h ... -> d l h (...)")
    x = NNop.flash_attention(q, k, v, pair; causal, kws...)
    return reshape(x, size(q))
end

sdpa(q, k, v; kws...) = naive_attention(q, k, v; kws...)
sdpa(q::AnyGPUArray, k::AnyGPUArray, v::AnyGPUArray; kws...) =
    flash_attention(q, k, v; kws...)

end
