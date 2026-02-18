"""
    RMSNorm(dim::Int; T=Float32, eps=1f-5, zero_centered=false)

Root Mean Square Layer Normalization. As used in Llama3.
"""
@concrete struct RMSNorm <: StatisticalNorm
    weight
    eps
    offset
end

function RMSNorm(dim::Int; T=Float32, eps=1f-5, zero_centered=false)
    weight = zero_centered ? zeros(T, dim) : ones(T, dim)
    offset = zero_centered ? one(T) : zero(T)
    RMSNorm(weight, T(eps), offset)
end

(norm::RMSNorm)(x) = Primitives.rms_norm(x, norm.weight; norm.eps, norm.offset)

function fuse((; weight, offset, eps)::RMSNorm, x)
    @lazy (weight + offset) * x / √($mean(abs2, x; dims=1) + eps)
end

function Base.show(io::IO, norm::RMSNorm)
    print(io, "RMSNorm(", length(norm.weight),
        ", eps=", repr(norm.eps))
    norm.offset != 0 && print(io, ", zero_centered=", true)
    print(io, ")")
end
