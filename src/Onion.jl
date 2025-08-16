module Onion

using BatchedTransformations
using ChainRulesCore
using ConcreteStructs
using Einops
using Flux
using LinearAlgebra

include("shared.jl")
export glut, zeros_like, ones_like

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
export reverse_tuple
export process_encoders
export process_decoders

include("FSQ.jl")
export FSQ
export chunk
export unchunk

end
