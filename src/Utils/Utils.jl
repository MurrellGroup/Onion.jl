module Utils

using ArrayInterface: restructure
using BFloat16s: BFloat16
using ChainRulesCore
using Einops
using LinearAlgebra

include("mul.jl")
export @mul!, mul, ᵀ

include("glut.jl")
export glut

include("reshapable.jl")
export reshapable, hardreshape

include("like.jl")
export like, zeros_like, ones_like, falses_like, trues_like

include("watmul.jl")
export watmul, ⨝

include("add_something.jl")
export ⊞

include("ofeltype.jl")
export ofeltype

include("masks.jl")
export causal_mask, self_att_padding_mask, cross_att_padding_mask

include("b16.jl")
export bf16

end
