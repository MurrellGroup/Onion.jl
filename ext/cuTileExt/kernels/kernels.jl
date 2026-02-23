using OnionStyle: ᵀ, →, i32, Optional

using cuTile:
    cuTile as ct,
    TileArray,
    TFloat32,
    BFloat16,
    Constant

using .ct.Experimental: autotune_launch, CartesianSpace

const TileVector{T} = TileArray{T,1}
const TileMatrix{T} = TileArray{T,2}
const TileArray3{T} = TileArray{T,3}
const TileArray4{T} = TileArray{T,4}
const TileArray5{T} = TileArray{T,5}

using DLFP8Types

#==============================================================
  ┌───────────────┬────────────┬───────────────┬────────────┐
  │    eltype     │ arithmetic │  tensorcore   │ accumulate │
  ├───────────────┼────────────┼───────────────┼────────────┤
  │ Float64       │ Float64    │ Float64       │ Float64    │
  ├───────────────┼────────────┼───────────────┼────────────┤
  │ Float32       │ Float32    │ TFloat32      │ Float32    │
  ├───────────────┼────────────┼───────────────┼────────────┤
  │ BFloat16      │ BFloat16   │ BFloat16      │ Float32    │
  ├───────────────┼────────────┼───────────────┼────────────┤
  │ Float16       │ Float16    │ Float16       │ Float16    │
  ├───────────────┼────────────┼───────────────┼────────────┤
  │ Float8_E4M3FN │ Float16    │ Float8_E4M3FN │ Float16    │
  ├───────────────┼────────────┼───────────────┼────────────┤
  │ Float8_E5M2   │ Float16    │ Float8_E5M2   │ Float16    │
  └───────────────┴────────────┴───────────────┴────────────┘
==============================================================#

arithmetic_type(T::Type) = T
arithmetic_type(::Type{TFloat32}) = Float32
arithmetic_type(::Type{Float8_E4M3FN}) = Float16
arithmetic_type(::Type{Float8_E5M2}) = Float16

tensorcore_type(T::Type) = T
tensorcore_type(::Type{Float32}) = TFloat32

accumulate_type(T::Type) = T
accumulate_type(::Type{TFloat32}) = Float32
accumulate_type(::Type{BFloat16}) = Float32
accumulate_type(::Type{Float16}) = Float16
accumulate_type(::Type{Float8_E4M3FN}) = Float16
accumulate_type(::Type{Float8_E5M2}) = Float16

include("attention/attention.jl")

include("feedforward/multihead.jl")

include("feedforward/swiglu.jl")

include("norm/layer_norm.jl")

include("norm/rms_norm.jl")

include("softmax.jl")
