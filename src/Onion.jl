module Onion

using ChainRulesCore
using ConcreteStructs
using Einops
using Flux
using LinearAlgebra
using NNlib

using Flux: @layer

abstract type Layer end
@layer Layer

export @concrete
export @layer

include("Utils/Utils.jl")
using .Utils
export glut
export like, zeros_like, ones_like, falses_like, trues_like
export watmul, ⨝
export causal_mask, self_att_padding_mask, cross_att_padding_mask
export bf16

const Maybe{T} = Union{T,Nothing}

include("Ops/Ops.jl")

include("ipa/ipa.jl")
include("miscellaneous/miscellaneous.jl")
include("norm/norm.jl")
include("connections/connections.jl")
include("convolution/convolution.jl")
include("positional-encoding/positional-encoding.jl")
include("transformers/transformers.jl")

end
