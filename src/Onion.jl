module Onion

using BatchedTransformations
using ChainRulesCore
using ConcreteStructs
using Einops
using Flux
using LinearAlgebra

include("shared.jl")
export glut

include("AdaLN.jl")
export AdaLN

include("RMSNorm.jl")
export RMSNorm

include("DyT.jl")
export DyT

include("StarGLU.jl")
export StarGLU

include("Ops.jl")
using .Ops: naive_attention, flash_attention

include("GQAttention.jl")
export Attention
export TransformerBlock
export AdaTransformerBlock
export DART
export self_att_padding_mask
export cross_att_padding_mask

include("RoPE.jl")
export RoPE

include("UNet.jl")
export GaussianFourierProjection
export TimeEmbedding
export ResidualBlock
export EncoderBlock
export DecoderBlock
export Bottleneck

include("FlexibleUNet.jl")
export FlexibleUNet
export reverse_tuple
export process_encoders
export process_decoders

include("FSQ.jl")
export FSQ
export chunk
export unchunk

include("IPAHelpers.jl")
export Framemover
export IPAblock
export CrossFrameIPA
export pair_encode

include("MultidimRoPE.jl")
export MultidimRoPE

include("STRING.jl")
export STRINGRoPE
export STRINGTransformerBlock
export AdaSTRINGTransformerBlock

end
