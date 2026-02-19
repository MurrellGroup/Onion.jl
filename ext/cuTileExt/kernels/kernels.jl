using OnionStyle: ᵀ, →
# XXX: i32 in OnionStyle
using GPUToolbox: i32

using cuTile:
    cuTile as ct,
    TileArray,
    TFloat32,
    Constant

using .ct.Experimental: autotune_launch, CartesianSpace

const TileVector{T} = TileArray{T,1}
const TileMatrix{T} = TileArray{T,2}
const TileArray3{T} = TileArray{T,3}
const TileArray4{T} = TileArray{T,4}
const TileArray5{T} = TileArray{T,5}

include("attention/attention.jl")

include("feedforward/multihead.jl")

include("feedforward/swiglu.jl")

include("norm/layer_norm.jl")

include("norm/rms_norm.jl")
