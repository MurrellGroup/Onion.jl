using Onion: Onion, TillitBackend, DefaultBackend
using OnionStyle: Optional
using ChainRulesCore: ChainRulesCore as CRC, NoTangent, unthunk
import ChainRulesCore: rrule

import Zygote

include("attention.jl")

include("fused_qknorm_rope.jl")

include("swiglu.jl")

include("norm/layer_norm.jl")

include("norm/rms_norm.jl")

include("norm/fused_add_rms_norm.jl")

include("softmax.jl")

include("linear.jl")

include("recurrent/deltanet.jl")

include("recurrent/fused_deltanet_decode.jl")

include("recurrent/causal_conv1d.jl")

include("recurrent/deltanet_sequence.jl")

include("recurrent/causal_conv1d_sequence.jl")
