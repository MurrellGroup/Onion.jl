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

LpNorm{p}(; dims=1, eps=1f-7) where p = LpNorm{p}(dims, eps)
LpNorm(p::Int; kws...) = LpNorm{p}(; kws...)

abs3(x) = abs2(x) * abs(x)
abs4(x) = abs2(abs2(x))

((; dims, eps)::LpNorm{1})(x) = x ./ (sum(abs, x; dims) .+ ofeltype(eps, x))
((; dims, eps)::LpNorm{2})(x) = x ./ (.√sum(abs2, x; dims) .+ ofeltype(eps, x))
((; dims, eps)::LpNorm{3})(x) = x ./ (.∛sum(abs3, x; dims) .+ ofeltype(eps, x))
((; dims, eps)::LpNorm{4})(x) = x ./ (.∜sum(abs4, x; dims) .+ ofeltype(eps, x))
((; dims, eps)::LpNorm{p})(x) where p = x ./ (sum(a -> abs(a^p), x; dims) .^ (1//p) .+ ofeltype(eps, x))

"""
    L2Norm(; dims=1, eps=1f-6)

Alias for [`LpNorm`](@ref) with `p=2`.
"""
const L2Norm = LpNorm{2}
