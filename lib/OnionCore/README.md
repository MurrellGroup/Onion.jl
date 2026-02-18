# OnionCore

Foundation types for the Onion neural network library.

## Exports

- **`Rules(; kws...)`** — Typed key-value container for propagating configuration through layer calls. Wraps a `NamedTuple` for static dispatch on contents. Supports `merge`, iteration, `get`, property access.

- **`Layer`** — Abstract type for neural network layers. Subtypes are callable structs that dispatch through `LayerStyle`.

- **`LayerStyle`** / **`EagerStyle`** — Controls how a layer is evaluated. Override `LayerStyle(::Type{MyLayer})` to customize. Default is `EagerStyle`, which calls `forward`.

- **`Backend`** / **`DefaultBackend`** — Abstract type hierarchy for compute backends. Backends are abstract types dispatched via `Type{<:Backend}`, not instances.

## Layer interface

```julia
struct MyLayer <: Layer
    ...
end

# default LayerStyle implements `forward`
forward(l::MyLayer, x) = ...

# construct layer
l = MyLayer(...)

# call forward
l(x)

# pass rules through
l(Rules(do_this = true), x)
```

`forward(layer, ::Rules, args...)` strips `Rules` by default — layers opt in to receiving them.
