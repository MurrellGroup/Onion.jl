"""
    RMSNorm(dim::Int; eps::T=1f-5)

Root Mean Square Layer Normalization. As used in Llama3.
"""
@concrete struct RMSNorm
    weight
    eps
end

Flux.@layer RMSNorm

RMSNorm(dim::Int; eps::T=1f-5) where T = RMSNorm(ones(T, dim), eps)

(norm::RMSNorm)(x) = Ops.rms_norm(x, norm.weight; norm.eps)
