module Onion

using Republic

# bring in all public names, and reexport all exported names
@republic reexport=true using OnionCore

using OnionStyle
using Rewrap: Keep, Split

using ConcreteStructs: @concrete
using ChainRulesCore: @ignore_derivatives

@republic import Optimisers: trainable

include("utils/utils.jl")

include("fuse.jl")
public fuse

include("backends.jl")
export DefaultBackend
export NNopBackend
export cuTileBackend

include("primitives/primitives.jl")
public Primitive
public backend, backend!, withbackend
public @primitive

include("layers/layers.jl")

function __init__()
    backend!(DefaultBackend())
end

end
