"""
    PNorm(p; dims=1, eps=1f-6)
    PNorm{p}(; dims=1, eps=1f-6)

A p-norm layer. This layer has no trainable parameters.

See also the [`L2Norm`](@ref) alias for `p=2`.
"""
@concrete struct PNorm{p}
    dims
    eps
end

@layer PNorm

PNorm{p}(; dims=1, eps=1f-6) where p = PNorm{p}(dims, eps)
PNorm(p::Int; kws...) = PNorm{p}(; kws...)

(norm::PNorm{1})(x) = x ./ (sum(abs, x; norm.dims) .+ ofeltype(norm.eps, x))
(norm::PNorm{2})(x) = x ./ (.√sum(abs2, x; norm.dims) .+ ofeltype(norm.eps, x))
(norm::PNorm{p})(x) where p = x ./ (sum(a -> abs(a)^p, x; norm.dims) .^ (1/p) .+ ofeltype(norm.eps, x))

"""
    L2Norm(; dims=1, eps=1f-6)

Alias for [`PNorm`](@ref) with `p=2`.
"""
const L2Norm = PNorm{2}
