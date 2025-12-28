abstract type AbstractConnections <: Layer end

include("skip.jl")
export SkipConnections, ResidualConnections

include("hyper.jl")
export GeneralizedHyperConnections

include("VirtualWidthNetwork.jl")
export VirtualWidthNetwork
