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
@concrete struct DART <: Layer
    transformer
end

function (dart::DART)(x::AbstractArray; pair_feats=nothing, kws...)
    h = rearrange(x, (:d, :K, :L, ..) --> (:d, (:K, :L), ..))
    r = size(x, 2)
    isnothing(pair_feats) || (pair_feats = repeat(pair_feats, einops"h ql kl ... -> h (r1 ql) (r2 kl) ...", r1=r, r2=r))
    return reshape(dart.transformer(h; pair_feats, kws...), size(x))
end