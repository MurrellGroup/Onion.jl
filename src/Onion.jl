module Onion

using Republic

# bring in all public names, and reexport all exported names
@republic reexport=true using OnionCore
@republic import OnionCore: LayerStyle, apply_with, forward

using OnionStyle

using ConcreteStructs: @concrete
using ChainRulesCore: @ignore_derivatives

using Rewrap

@republic import Optimisers: trainable

include("utils/utils.jl")

include("fuse.jl")
public fuse

include("backends/backends.jl")
public backend, backend!, withbackend
export DefaultBackend
export NNkernelsBackend
export TillitBackend

include("primitives/primitives.jl")

include("layers/layers.jl")

function __init__()
    backend!(DefaultBackend())
end

end
