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

const Maybe{T} = Union{T, Nothing}

fix_mask(mask) = mask
fix_mask(mask::AbstractArray{<:Any, 3}) = rearrange(mask, einops"kl ql b -> kl ql 1 b")

function naive_attention(
    q::AbstractArray{T}, k::AbstractArray{T}, v::AbstractArray{T};
    mask=false,
) where T<:Number
    d = size(q, 1)
    kT = rearrange(k, einops"d kl ... -> kl d ...")
    a = kT ⊠ q ./ √T(d) .+ fix_mask(mask)
    return v ⊠ softmax(a)
end

function flash_attention(q::AnyGPUArray, k::AnyGPUArray, v::AnyGPUArray; kws...)
    NNop.flash_attention(q, k, v; kws...)
end

end