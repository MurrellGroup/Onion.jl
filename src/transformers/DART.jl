"""
    DART(transformer; mask=:causal)

"Doubly Auto-Regressive Transformer" (DART) is a convenience layer wrapping a
transformer block that can be used to model auto-regressive data represented
along two dimensions.

!!! note
    The mask acts on the flattened tokens sequence.

# Examples

```julia
julia> dart = DART(TransformerBlock(64, 8));

julia> x = randn(Float32, 64, 4, 20);

julia> dart(x) |> size
(64, 4, 20)
```
"""
@concrete struct DART
    transformer
end

Flux.@layer DART

function (dart::DART)(x::AbstractArray; mask=:causal, kws...)
    h = rearrange(x, (:d, :K, :L, ..) --> (:d, (:K, :L), ..))
    mask === :causal && (mask = causal_mask(h))
    return reshape(dart.transformer(h; mask, kws...), size(x))
end
