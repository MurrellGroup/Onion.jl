module Onion

using Republic

# bring in all public names, and reexport all exported names
@republic reexport=true using OnionCore

using OnionStyle

using ConcreteStructs: @concrete
using ChainRulesCore: @ignore_derivatives
@republic using NNlib: swish

include("utils/utils.jl")

include("fuse.jl")
public fuse

include("Primitives/Primitives.jl")
@republic using .Primitives

include("backends.jl")
public DefaultBackend
public NNopBackend
public cuTileBackend

include("primitives/primitives.jl")

include("layers/layers.jl")

function __init__()
    backend!(DefaultBackend())
end

end
