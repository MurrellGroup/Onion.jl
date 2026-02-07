module Onion

using Base.Broadcast: materialize

using ChainRulesCore
using ConcreteStructs
using Einops
using Flux
using LinearAlgebra
using NNlib
using Rewrap
using Statistics: mean

const Maybe{T} = Union{T,Nothing}

include("Utils/Utils.jl")
using .Utils
import .Utils: split
export split_axis
export glut
export like, zeros_like, ones_like, falses_like, trues_like
export watmul, ⨝
export self_att_padding_mask
export cross_att_padding_mask
export causal_mask
export bf16

include("Layer.jl")
export @concrete
export @layer

include("Ops/Ops.jl")

include("ipa/ipa.jl")
include("miscellaneous/miscellaneous.jl")
include("normalization/normalization.jl")
include("connections/connections.jl")
include("convolution/convolution.jl")
include("positional-encoding/positional-encoding.jl")
include("transformers/transformers.jl")
include("protein/protein.jl")

end
