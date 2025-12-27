"""
    Modulator(in_dim => out_dim; σ=sigmoid, op=*)

Takes an input `Y` and a conditioning input `X` and applies a gate to `Y` based on `X`.

See [Gated Attention for Large Language Models](https://arxiv.org/pdf/2505.06708)

# Examples

```jldoctest
julia> gate = Modulator(32 => 64);

julia> Y = randn(Float32, 64);

julia> X = randn(Float32, 32);

julia> gate(Y, X) |> size
(64,)
```
"""
@concrete struct Modulator
    W; σ; op; shape
end

@layer Modulator

function Modulator(
    (in, out)::Pair{Int,Int}, σ=sigmoid;
    op=*, bias=false, shape=nothing, init=Flux.kaiming_uniform
)
    shape = isnothing(shape) ? out : shape
    prod(shape) == out || throw(DimensionMismatch("prod(shape) must be equal to out"))
    W = Linear(in => out; bias, init)
    return Modulator(W, σ, op, shape)
end

function Modulator((in, shape)::Pair{Int,<:Tuple{Vararg{Int}}}; kws...)
    return Modulator(in => prod(shape); shape, kws...)
end

function (m::Modulator)(Y, X)
    W, σ, * = m.W, m.σ, m.op
    WX = reshape(W(X), m.shape..., size(X)[2:end]...)
    Y′ = Y .* σ.(WX)
    return Y′
end
