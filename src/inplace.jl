struct InplaceStyle <: LayerStyle end

# Same style → call native interface
apply_with(::InplaceStyle, ::InplaceStyle, layer::Layer, r::Rules, args...; kws...) =
    inplace(layer, r, args...; kws...)

# Cross-style bridging
apply_with(::EagerStyle, ::InplaceStyle, layer::Layer, r::Rules, args...; kws...) =
    forward(layer, r, args...; kws...)
apply_with(::InplaceStyle, ::EagerStyle, layer::Layer, r::Rules, args...; kws...) =
    forward(layer, r, args...; kws...)

# Default: strip rules for layers that don't accept them
inplace(layer::Layer, ::Rules, args...; kws...) = inplace(layer, args...; kws...)

# Missing implementation errors
inplace(::L, args...; kws...) where L<:Layer =
    error("$(nameof(L)) does not implement inplace")
