using Base.Broadcast: materialize

struct FusedStyle <: LayerStyle end

# Same style → call native interface
apply_with(::FusedStyle, ::FusedStyle, layer::Layer, r::Rules, args...; kws...) =
    fuse(layer, r, args...; kws...)

# Cross-style bridging
apply_with(::EagerStyle, ::FusedStyle, layer::Layer, r::Rules, args...; kws...) =
    materialize(fuse(layer, r, args...; kws...))
apply_with(::FusedStyle, ::EagerStyle, layer::Layer, r::Rules, args...; kws...) =
    forward(layer, r, args...; kws...)

# Default: strip rules for layers that don't accept them
fuse(layer::Layer, ::Rules, args...; kws...) = fuse(layer, args...; kws...)

# Missing implementation errors
fuse(::L, args...; kws...) where L<:Layer =
    error("$(nameof(L)) does not implement fuse")
