using Onion: Onion, cuTileBackend, DefaultBackend
using ChainRulesCore: ChainRulesCore as CRC, NoTangent, unthunk

import Zygote

function (p::Onion.Primitive)(::cuTileBackend, args...; kws...)
    return p(DefaultBackend(), args...; kws...)
end

function (p::Onion.Primitive)(::cuTileBackend, r::Onion.Rules, args...; kws...)
    return p(cuTileBackend(), args...; kws...)
end

include("attention/attention.jl")

include("feedforward/multihead.jl")

include("feedforward/swiglu.jl")

include("norm/layer_norm.jl")

include("norm/rms_norm.jl")

include("softmax.jl")

include("linear.jl")

include("newton_schulz.jl")
