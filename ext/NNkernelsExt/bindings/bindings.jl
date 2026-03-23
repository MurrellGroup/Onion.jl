using Onion: Onion, Primitive, NNkernelsBackend, DefaultBackend

function (p::Primitive)(::NNkernelsBackend, args...; kws...)
    return p(DefaultBackend(), args...; kws...)
end

using Rewrap: Keep, Split, (..)

include("rms_norm.jl")
include("layer_norm.jl")
include("softmax.jl")
include("attention.jl")
