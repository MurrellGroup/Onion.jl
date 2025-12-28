"""
    With(wrapper, layer)

Wrap `layer` with `wrapper`, calling `wrapper(layer, args...; kws...)`.

# Examples

```jldoctest
julia> model = With(GHC(3, 2), Linear(8 => 8));

julia> x = randn(Float32, 12, 5);

julia> model(x) |> size
(12, 5)
```
"""
@concrete struct With <: Layer
    wrapper
    layer
end

function ((; wrapper, layer)::With)(args...; kws...)
    return wrapper(layer, args...; kws...)
end