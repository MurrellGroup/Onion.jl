module Onion

using Flux
using LinearAlgebra
using BatchedTransformations
using ConcreteStructs
using Einops

include("shared.jl")
include("AdaLN.jl")
include("RMSNorm.jl")
include("DyT.jl")
include("StarGLU.jl")
include("GQAttention.jl")
include("RoPE.jl")
include("UNet.jl")
include("FlexibleUNet.jl")
include("FSQ.jl")
include("IPAHelpers.jl")

export
    #shared:
    glut,
    #layers:
    AdaLN,
    RMSNorm,
    DyT,
    StarGLU,
    GQAttention,
    RoPE,
    TransformerBlock,
    Attention,
    FSQ,
    Framemover,
    IPAblock,
    CrossFrameIPA,
    pair_encode,
    chunk,
    unchunk,
    self_att_padding_mask,
    cross_att_padding_mask,
    # UNet components:
    GaussianFourierProjection,
    TimeEmbedding,
    ResidualBlock,
    EncoderBlock,
    DecoderBlock,
    Bottleneck,
    FlexibleUNet,
    # UNet helper functions:
    reverse_tuple,
    process_encoders,
    process_decoders
end
