"""
    LpNorm(p; dims=1, eps=1f-6)
    LpNorm{p}(; dims=1, eps=1f-6)

A p-norm layer. This layer has no trainable parameters.

See also the [`L2Norm`](@ref) alias for `p=2`.
"""
@concrete struct LpNorm{p}
    dims
    eps
end

@layer LpNorm

LpNorm{p}(; dims=1, eps=1f-6) where p = LpNorm{p}(dims, eps)
LpNorm(p::Int; kws...) = LpNorm{p}(; kws...)

(norm::LpNorm{1})(x) = x ./ (sum(abs, x; norm.dims) .+ ofeltype(norm.eps, x))
(norm::LpNorm{2})(x) = x ./ (.√sum(abs2, x; norm.dims) .+ ofeltype(norm.eps, x))
(norm::LpNorm{p})(x) where p = x ./ (sum(a -> abs(a)^p, x; norm.dims) .^ (1/p) .+ ofeltype(norm.eps, x))

"""
    L2Norm(; dims=1, eps=1f-6)

Alias for [`LpNorm`](@ref) with `p=2`.
"""
const L2Norm = LpNorm{2}
