# Primitives

`Onion.Primitives` provides backend-dispatched primitives:

`linear`, `rms_norm`, `layer_norm`, `softmax`, `attention`, `glu_ffn`, `multihead_ffn`, `rotary_pos_emb`, `combine_projections`, `quintic_newton_schulz`

Each primitive is a callable singleton (`<: Primitive <: Function`) that dispatches through a concrete `Backend` instance.

## Declaring and implementing primitives

`@primitive` declares a primitive:

```julia
@primitive rms_norm
```

This creates a singleton struct and a const instance:

```julia
struct primitive_rms_norm <: Primitive end
const rms_norm = primitive_rms_norm()
```

Implement a primitive by defining a method on the singleton for a specific backend:

```julia
rms_norm(::DefaultBackend, x::AbstractArray, w::AbstractVector; eps=1f-5) = ...
```

## Backends

Backends are concrete structs, all directly subtyping `Backend`:

```
Backend (abstract root)
├── DefaultBackend   (CPU/GPU fallback implementations)
├── NNopBackend      (KernelAbstractions-based — FluxML/NNop.jl)
└── cuTileBackend    (cuTile DSL-generated CUDA kernels)
```

There is no abstract subtype hierarchy. Instead, backends define **explicit fallback methods**:

```julia
# In ext/cuTileExt — unimplemented primitives delegate to DefaultBackend:
(p::Primitive)(::cuTileBackend, args...; kws...) = p(DefaultBackend(), args...; kws...)

# In ext/NNopExt — same pattern:
(p::Primitive)(::NNopBackend, args...; kws...) = p(DefaultBackend(), args...; kws...)
```

A backend only needs to implement the primitives it specializes; the rest fall through via the explicit fallback.

## Dispatch chain

Every primitive call flows through the same chain:

```julia
p(args...)                        # 1. wrap in empty Rules
p(rules::Rules, args...)          # 2. extract backend from rules
p(backend, rules::Rules, args...) # 3. strip Rules (default) or use them
p(backend, args...)               # 4. → backend-specific implementation
```

Implementations that need `Rules` can accept them explicitly; otherwise they are stripped automatically.

## Backend selection

Three mechanisms, in order of precedence:

1. **`Rules`** (explicit, type-stable): `p(Rules(backend=cuTileBackend()), x)`
2. **`withbackend` scope** (task-local): `withbackend(cuTileBackend()) do ... end`
3. **`backend!`** (global): `backend!(cuTileBackend())`

`DefaultBackend()` is set as the global backend in `Onion.__init__()`.

### Rules options

```julia
# Direct backend instance:
Rules(backend=cuTileBackend())

# Per-primitive selection — any callable (receives the primitive, returns a Backend):
Rules(backend=Returns(cuTileBackend()))    # same backend for all
Rules(backend=my_selector_function)        # custom dispatch on primitive

# OnionStyle.@staticmap helper — compile-time dispatch on primitive identity:
Rules(backend = @staticmap attention => cuTileBackend(), _ => DefaultBackend())

# Block syntax with grouping:
Rules(backend = @staticmap begin
    _ => cuTileBackend()
    attention => NNopBackend()
    {rms_norm, softmax} => DefaultBackend()
end)
```

## Adding a backend

Define a concrete backend struct (subtyping `Backend`) and provide implementations via a package extension:

```julia
# In Onion (src/backends.jl):
struct MyBackend <: Backend end

# In ext/MyBackendExt:

# Fallback — delegate unimplemented primitives:
(p::Onion.Primitive)(::MyBackend, args...; kws...) = p(DefaultBackend(), args...; kws...)

# Strip Rules for the fallback too:
(p::Onion.Primitive)(::MyBackend, r::Onion.Rules, args...; kws...) = p(MyBackend(), args...; kws...)

# Override specific primitives:
Onion.attention(::MyBackend, q, k, v; causal) = my_fast_attention(q, k, v; causal)
```
