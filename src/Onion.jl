module Onion

using BatchedTransformations
using ChainRulesCore
using ConcreteStructs
using Einops
using Flux
using LinearAlgebra
using NNlib

using Flux: @layer

include("shared.jl")
export glut
export like, zeros_like, ones_like, falses_like, trues_like
export watmul, ⨝

include("BlockDense.jl")
export BlockDense

include("masks.jl")
export self_att_padding_mask
export cross_att_padding_mask
export causal_mask

include("Ops/Ops.jl")

include("norm/norm.jl")

include("StarGLU.jl")
export StarGLU

include("transformers/transformers.jl")

include("embedding/embedding.jl")

include("InvariantPointAttention/InvariantPointAttention.jl")

include("DyT.jl")
export DyT

include("UNet/UNet.jl")
export GaussianFourierProjection
export TimeEmbedding
export ResidualBlock
export EncoderBlock
export DecoderBlock
export Bottleneck

include("UNet/FlexibleUNet.jl")
export FlexibleUNet
export process_encoders
export process_decoders

include("FSQ.jl")
export FSQ
export chunk
export unchunk

end
