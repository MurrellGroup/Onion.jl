"""
    Layer

Abstract type for all neural network layers. Subtypes are callable structs.
"""
abstract type Layer end

# --- LayerStyle dispatch ---

"""
    LayerStyle

Abstract type for layer evaluation styles. Determines how a layer is called.

Override for a layer type:

    LayerStyle(::Type{MyLayer}) = MyStyle()
"""
abstract type LayerStyle end

LayerStyle(::T) where T<:Layer = LayerStyle(T)
LayerStyle(::Type{<:Layer}) = EagerStyle()

struct EagerStyle <: LayerStyle end

# Entry points
function (layer::Layer)(args...; backend=nothing, kws...)
    r = isnothing(backend) ? Rules() : Rules(; backend)
    apply(EagerStyle(), layer, r, args...; kws...)
end

(layer::Layer)(r::Rules, args...; kws...) = apply(EagerStyle(), layer, r, args...; kws...)

# Style dispatch
apply(style::LayerStyle, layer::Layer, r::Rules, args...; kws...) =
    apply_with(style, LayerStyle(layer), layer, r, args...; kws...)

# Same style → call native interface
apply_with(::EagerStyle, ::EagerStyle, layer::Layer, r::Rules, args...; kws...) =
    forward(layer, r, args...; kws...)

# Default: strip rules for layers that don't accept them
forward(layer::Layer, ::Rules, args...; kws...) = forward(layer, args...; kws...)

# Missing implementation errors
forward(::L, args...; kws...) where L<:Layer =
    error("$(nameof(L)) does not implement forward")

# --- Unicode properties ---

const UNICODE_PROPERTY_MAP = (
    :α => :alpha,   :β => :beta,    :γ => :gamma,
    :δ => :delta,   :ε => :epsilon, :ζ => :zeta,
    :η => :eta,     :θ => :theta,   :ι => :iota,
    :κ => :kappa,   :λ => :lambda,  :μ => :mu,
    :ν => :nu,      :ξ => :xi,      :π => :pi,
    :ρ => :rho,     :σ => :sigma,   :τ => :tau,
    :υ => :upsilon, :φ => :phi,     :χ => :chi,
    :ψ => :psi,     :ω => :omega,
)

function withname(layer::Layer, s::Symbol)
    i = findfirst(==(s), first.(UNICODE_PROPERTY_MAP))
    unicode = i isa Int ?
        last(UNICODE_PROPERTY_MAP[i]) : nothing
    unicode isa Symbol ?
        getfield(layer, unicode) : throw(FieldError(typeof(layer), s))
end

function withunicode(layer::Layer, s::Symbol)
    i = findfirst(==(s), last.(UNICODE_PROPERTY_MAP))
    name = i isa Int ?
        first(UNICODE_PROPERTY_MAP[i]) : nothing
    name isa Symbol ?
        getfield(layer, name) : withname(layer, s)
end

function Base.getproperty(layer::Layer, s::Symbol)
    hasfield(typeof(layer), s) ? getfield(layer, s) : withunicode(layer, s)
end
