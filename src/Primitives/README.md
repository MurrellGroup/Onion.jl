# Primitives

`Onion.Primitives` provides backend-dispatched primitives (`rms_norm`, `layer_norm`, `softmax`, `attention`, `glu_ffn`, `multihead_ffn`).

Each primitive is a callable singleton (`<: Primitive <: Function`) that dispatches through a `Backend` type hierarchy.

## Declaring and implementing primitives

`@primitive` declares a primitive. `@impl` defines an implementation for a backend:

```julia
@primitive rms_norm

@impl DefaultBackend function rms_norm(x::AbstractArray; kws...)
    # default implementation
end
```

Backends form a type hierarchy — implementations are inherited by subtypes:

```
Backend (abstract root — no implementations, errors on use)
└── DefaultBackend (default CPU/GPU implementations)
    └── TileBackend (OnionTile — overrides specific primitives)
```

A backend only needs to override the primitives it specializes; the rest fall through to the parent.

## Dispatch chain

Every primitive call flows through the same chain:

```julia
p(args...)                        # wrap in empty Rules
p(rules::Rules, args...)          # extract backend, keep Rules
p(backend, rules::Rules, args...) # strip Rules (default) or use them
p(backend, args...)               # → backend-specific implementation
```

Implementations that need `Rules` can accept them explicitly; otherwise they are stripped automatically.

## Backend selection

Three mechanisms, in order of precedence:

1. **`Rules`** (explicit, type-stable): passed through layers
2. **`withbackend` scope** (task-local): `withbackend(TileBackend) do ... end`
3. **`backend!`** (global): `backend!(TileBackend)` / `resetbackend!()`

When no backend is specified, `DefaultBackend` is used.

### Rules options

```julia
# Convenient — one dynamic dispatch per layer call:
Rules(backend=TileBackend)

# Type-stable:
Rules(backend=Val(TileBackend))

# Per-primitive selection — any callable:
Rules(backend=Returns(TileBackend))       # same backend for all
Rules(backend=my_selector_function)       # dispatch on primitive

# OnionStyle.@staticmap helper:
# Create a static dict-like function with fallback:
Rules(backend = @staticmap attention => TileBackend, _ => DefaultBackend)

# or with block syntax:
Rules(backend = @staticmap begin
    _ => TileBackend
    attention => NNopBackend
    {rms_norm, softmax} => DefaultBackend
end)
```

## Adding a backend

Define a backend type (in OnionCore's hierarchy) and provide implementations via a package extension:

```julia
# In OnionTile:
abstract type TileBackend <: DefaultBackend end

# In OnionTileOnionExt:
@impl TileBackend function Onion.Primitives.attention(q, k, v; causal)
    OnionTile.mha(q, k, v; causal)
end
```
