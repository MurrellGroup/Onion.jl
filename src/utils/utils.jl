using ChainRulesCore
using Einops
using LinearAlgebra

"""
    ofeltype(v::Number, x::AbstractArray{T}) where T = convert(T, v)

Convert `v` to type `T`.
"""
ofeltype(v::Number, x::AbstractArray{T}) where T = convert(T, v)

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
