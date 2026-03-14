using Onion: Onion, Primitive, NNopBackend, DefaultBackend

function (p::Primitive)(::NNopBackend, args...; kws...)
    return p(DefaultBackend(), args...; kws...)
end

using Rewrap: Keep, Split, (..)

include("rms_norm.jl")
include("layer_norm.jl")
include("softmax.jl")
include("attention.jl")
