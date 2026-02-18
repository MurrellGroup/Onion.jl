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
