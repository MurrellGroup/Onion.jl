using Onion: Onion, NNopBackend, DefaultBackend
using Onion: Primitive, @impl

@impl NNopBackend function (p::Primitive)(args...; kws...)
    return p(DefaultBackend(), args...; kws...)
end

include("rms_norm.jl")
include("layer_norm.jl")
include("softmax.jl")
include("attention.jl")
