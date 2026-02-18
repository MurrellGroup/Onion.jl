using Onion: Onion, cuTileBackend, DefaultBackend
using Onion: Primitive, @impl

@impl cuTileBackend function (p::Primitive)(args...; kws...)
    return p(DefaultBackend(), args...; kws...)
end

include("attention.jl")
include("glu.jl")
include("multihead.jl")
