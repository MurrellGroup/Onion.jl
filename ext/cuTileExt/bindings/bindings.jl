using Onion: Onion, cuTileBackend, DefaultBackend
using ChainRulesCore: ChainRulesCore as CRC, NoTangent, unthunk

import Zygote

function (p::Onion.Primitive)(::cuTileBackend, args...; kws...)
    return p(DefaultBackend(), args...; kws...)
end

# TODO: make primitives mutating, and have interfaces for the mutating interface?
# or use GPUArrays.AllocCache, but that's limiting

include("attention/attention.jl")

include("attention/fused_qknorm_rope.jl")

include("feedforward/multihead.jl")

include("feedforward/swiglu.jl")

include("norm/layer_norm.jl")

include("norm/rms_norm.jl")

include("norm/fused_add_rms_norm.jl")

include("softmax.jl")

include("linear.jl")

include("newton_schulz.jl")

include("recurrent/deltanet.jl")

include("recurrent/fused_deltanet_decode.jl")

include("recurrent/causal_conv1d.jl")

include("recurrent/deltanet_sequence.jl")

include("recurrent/causal_conv1d_sequence.jl")
