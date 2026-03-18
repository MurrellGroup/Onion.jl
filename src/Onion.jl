module Onion

using Republic

# TODO: alternate precision interface trait

# bring in all public names, and reexport all exported names
@republic reexport=true using OnionCore
@republic import OnionCore: LayerStyle, apply_with, forward

using OnionStyle

using ConcreteStructs: @concrete
using ChainRulesCore: @ignore_derivatives

@republic import Optimisers: trainable

include("utils/utils.jl")

include("fuse.jl")
public fuse

include("decode.jl")
public decode

include("backends.jl")
export DefaultBackend
export NNopBackend
export cuTileBackend

include("primitives/primitives.jl")
public Primitive, @primitive
public backend, backend!, withbackend

include("layers/layers.jl")

function __init__()
    backend!(DefaultBackend())
end

end
