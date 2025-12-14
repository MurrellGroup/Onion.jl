"""
    RMSNorm(dim::Int; T=Float32, eps=1f-5, zero_centered=false)

Root Mean Square Layer Normalization. As used in Llama3.
"""
@concrete struct RMSNorm
    weight
    eps
    offset
end

@layer RMSNorm

function RMSNorm(dim::Int; T=Float32, eps=1f-5, zero_centered=false)
    weight = zero_centered ? zeros(T, dim) : ones(T, dim)
    offset = zero_centered ? one(T) : zero(T)
    RMSNorm(weight, T(eps), offset)
end

(norm::RMSNorm)(x) = Ops.rms_norm(x, norm.weight; norm.eps, norm.offset)
