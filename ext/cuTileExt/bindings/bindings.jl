using Onion: Onion, cuTileBackend, DefaultBackend
using ChainRulesCore: ChainRulesCore as CRC, NoTangent, unthunk
import ChainRulesCore: rrule

import Zygote

# Note: no generic Primitive fallback here — it creates ambiguity with
# the @primitive dispatch chain `(b::Backend, r::Rules, args...)`.
# All used primitives have explicit cuTileBackend methods below.
# Unimplemented ones fall through to `(b::Backend, ...)` defaults,
# which work on CuArrays via GPUArrays broadcasting.

include("attention/attention.jl")

include("attention/fused_qknorm_rope.jl")

include("feedforward/multihead.jl")

include("feedforward/swiglu.jl")

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
