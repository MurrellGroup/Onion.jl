using ChainRulesCore
using Einops
using LinearAlgebra
using Random

unval(::Val{x}) where {x} = x
unval(x) = x

with_default_rng(init) = (args...; kws...) -> init(Random.default_rng(), args...; kws...)

"""
    ofeltype(v::Number, ::AbstractArray{T}) where T = convert(T, v)

Convert `v` to type `T`.
"""
ofeltype(v::Number, ::AbstractArray{T}) where T = convert(T, v)

macro constprop(expr::Expr)
    esc(:(Base.@constprop :aggressive $expr))
end

include("glut.jl")
export glut

include("splitaxis.jl")
export splitaxis

include("lazy.jl")
export @lazy

include("like.jl")
export like, zeros_like, ones_like, falses_like, trues_like

include("watmul.jl")
export watmul, ⨝

include("masks.jl")
export self_att_padding_mask
export cross_att_padding_mask
export causal_mask
