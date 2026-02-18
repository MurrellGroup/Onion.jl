using Onion: Onion, NNopBackend, DefaultBackend
using Onion: Primitive, @impl

@impl NNopBackend function (p::Primitive)(args...; kws...)
    return p(DefaultBackend(), args...; kws...)
end

include("attention.jl")