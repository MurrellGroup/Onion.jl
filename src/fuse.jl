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
fuse(layer::Layer, r::Rules, args...; kws...) = fuse(layer, args...; kws...)

# Default: inject empty rules for layers that only define the Rules variant.
# A task-local identity guard detects the cycle when neither variant is specialized.
function fuse(layer::Layer, args...; kws...)
    seen = get(task_local_storage(), :_onion_fuse_seen, nothing)
    layer === seen && throw(MethodError(fuse, (layer, args...)))
    task_local_storage(:_onion_fuse_seen, layer)
    try
        return fuse(layer, Rules(), args...; kws...)
    finally
        task_local_storage(:_onion_fuse_seen, seen)
    end
end
