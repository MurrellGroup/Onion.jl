abstract type AbstractConnection <: Layer end

include("skip.jl")
export SkipConnection, ResidualConnection

include("hyper.jl")
export GeneralizedHyperConnection, GHC

include("VirtualWidthNetwork.jl")
export VirtualWidthNetwork
