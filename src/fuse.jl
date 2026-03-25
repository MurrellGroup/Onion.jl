using Base.Broadcast: materialize

struct FusedStyle <: LayerStyle end

# Same style → call native interface
apply_with(::FusedStyle, ::FusedStyle, layer::Layer, args...; kws...) =
    fuse(layer, args...; kws...)

# Cross-style bridging
apply_with(::EagerStyle, ::FusedStyle, layer::Layer, args...; kws...) =
    materialize(fuse(layer, args...; kws...))
apply_with(::FusedStyle, ::EagerStyle, layer::Layer, args...; kws...) =
    forward(layer, args...; kws...)

fuse(layer::Layer, args...; kws...) = throw(MethodError(fuse, (layer, args...)))
