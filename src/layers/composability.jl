@concrete struct Composed <: Layer
    inner
    outer
end

function ((; inner, outer)::Composed)(args...; kws...)
    return outer(inner(args...; kws...))
end


"""
    With(wrapper, layer)

Wrap `layer` with `wrapper`, calling `wrapper(layer, args...; kws...)`.

Equivalent to `Base.Fix1(wrapper, layer)`.

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


"""
    AdaAffine(f, dim, cond_dim; zero_init, bias = false)

Adaptive Affine layer, using a secondary conditioning input to scale and shift the output
of the wrapped layer.

Equivalent to [`AdaLN`](@ref) when wrapping a [`LayerNorm`](@ref) layer,
and [AdaLN-Zero]() when initialized with `zero_init=true`.
"""
struct AdaAffine{F,T} <: Layer
    f::F
    scale_proj::T
    shift_proj::T
end

function AdaAffine(
    f, dim::Integer, cond_dim::Integer;
    bias = false, init_zero = false
)
    scale_proj = Linear(cond_dim => dim; bias)
    shift_proj = Linear(cond_dim => dim; bias)
    if init_zero
        scale_proj.weight .= 0
        shift_proj.weight .= 0
        @assert iszero(scale_proj.bias) "expected scale_proj.bias to be zero"
        @assert iszero(shift_proj.bias) "expected shift_proj.bias to be zero"
    end
    return AdaAffine(f, scale_proj, shift_proj)
end

LayerStyle(::Type{<:AdaAffine}) = FusedStyle()

function fuse((; f, scale_proj, shift_proj)::AdaAffine, x, cond)
    γ = scale_proj(cond)
    β = shift_proj(cond)
    return @lazy (1 + γ) * $fuse(f, x) + β
end


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
@concrete struct Modulator <: Layer
    W; σ; op; shape
end

function Modulator(
    (in, out)::Pair{Int,Int}, σ=NNlib.sigmoid;
    op=*, bias=false, shape=nothing, kws...
)
    shape = isnothing(shape) ? out : shape
    prod(shape) == out || throw(DimensionMismatch("prod(shape) must be equal to out"))
    W = Linear(in => out; bias, kws...)
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
