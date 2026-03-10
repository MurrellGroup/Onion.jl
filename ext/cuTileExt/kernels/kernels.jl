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

tensorcore_type(T::Type) = T
tensorcore_type(::Type{Float32}) = TFloat32

accumulate_type(T::Type) = T
accumulate_type(::Type{TFloat32}) = Float32
accumulate_type(::Type{BFloat16}) = Float32
accumulate_type(::Type{Float16}) = Float16

@inline function swizzle_2d(M, N, tm, tn, GROUP_SIZE_M, bid)
  num_bid_m = cld(M, Int32(tm))
  num_bid_n = cld(N, Int32(tn))
  num_bid_in_group = Int32(GROUP_SIZE_M) * num_bid_n
  group_id = fld(bid, num_bid_in_group)
  first_bid_m = group_id * Int32(GROUP_SIZE_M)
  group_size_m = min(num_bid_m - first_bid_m, Int32(GROUP_SIZE_M))
  bid_m = first_bid_m + rem(bid, group_size_m)
  bid_n = fld(rem(bid, num_bid_in_group), group_size_m)
  return bid_m, bid_n
end

include("attention/attention.jl")

include("feedforward/multihead.jl")

include("feedforward/swiglu.jl")

include("norm/layer_norm.jl")

include("norm/rms_norm.jl")

include("softmax.jl")

include("linear.jl")

include("newton_schulz.jl")
