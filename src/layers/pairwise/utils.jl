using Random

# ──── Training mode toggle ────

const _TRAINING = Ref(false)

training_mode() = _TRAINING[]
training_mode!(flag::Bool) = (_TRAINING[] = flag)

# ──── SharedDropout ────

@concrete struct SharedDropout <: Layer
    rate::Float32
    broadcast_dims::Vector{Int}  # dims set to size 1 in the mask
end

SharedDropout(rate::Real, dims) =
    SharedDropout(Float32(rate), isa(dims, Int) ? [dims] : collect(dims))

function (m::SharedDropout)(x)
    (m.rate == 0f0 || !_TRAINING[]) && return x
    T = eltype(x)
    shape = collect(size(x))
    for d in m.broadcast_dims
        shape[d] = 1
    end
    mask = rand_like(x, T, shape...)
    keep = one(T) - T(m.rate)
    scale = one(T) / keep
    return x .* ifelse.(mask .>= T(m.rate), scale, zero(T))
end

# ──── Dropout mask generators ────

function get_dropout_mask(
    dropout::Real,
    z::AbstractArray;
    columnwise::Bool=false,
    training::Bool=training_mode(),
)
    T = eltype(z)
    eff_dropout = T(dropout) * (training ? one(T) : zero(T))
    eff_dropout ≈ 0 && return one(T)
    if columnwise
        v = z[1:1, 1:1, :, :]
    else
        v = z[1:1, :, 1:1, :]
    end
    d = rand_like(z, Float32, size(v)...) .> eff_dropout
    scale = T(1 / (1 - eff_dropout))
    return T.(d) .* scale
end

get_dropout_mask_columnwise(dropout::Real, z::AbstractArray; kws...) =
    get_dropout_mask(dropout, z; columnwise=true, kws...)

get_dropout_mask_rowise(dropout::Real, z::AbstractArray; kws...) =
    get_dropout_mask(dropout, z; columnwise=false, kws...)

# ──── Weight initialization functions ────

const _TRUNCNORM_STD = 0.8796256610342398

function _calculate_fan(weight_shape::NTuple{2,Int}, fan::Symbol)
    fan_out, fan_in = weight_shape
    fan === :fan_in  && return fan_in
    fan === :fan_out && return fan_out
    fan === :fan_avg && return (fan_in + fan_out) / 2
    error("Invalid fan option: $fan")
end

function trunc_normal_init!(weights; scale=1.0, fan::Symbol=:fan_in)
    shape = size(weights)
    length(shape) == 2 || error("trunc_normal_init! expects 2D weight, got size $(shape)")
    f = _calculate_fan(shape, fan)
    std = sqrt(scale / max(1, f)) / _TRUNCNORM_STD
    rng = Random.default_rng()
    for i in eachindex(weights)
        v = randn(rng)
        while v < -2 || v > 2
            v = randn(rng)
        end
        weights[i] = v * std
    end
    return weights
end

lecun_normal_init!(weights) = trunc_normal_init!(weights; scale=1.0)
he_normal_init!(weights) = trunc_normal_init!(weights; scale=2.0)

function glorot_uniform_init!(weights)
    shape = size(weights)
    length(shape) == 2 || error("glorot_uniform_init! expects 2D weight, got size $(shape)")
    fan_out, fan_in = shape
    limit = sqrt(6 / (fan_in + fan_out))
    rng = Random.default_rng()
    weights .= (rand(rng, eltype(weights), size(weights)) .* (2 * limit) .- limit)
    return weights
end

final_init!(weights) = (weights .= zero(eltype(weights)); weights)
gating_init! = final_init!

bias_init_zero!(bias) = (bias .= zero(eltype(bias)); bias)
bias_init_one!(bias) = (bias .= one(eltype(bias)); bias)

function normal_init!(weights)
    shape = size(weights)
    length(shape) == 2 || error("normal_init! expects 2D weight, got size $(shape)")
    _, fan_in = shape
    std = inv(sqrt(fan_in))
    rng = Random.default_rng()
    weights .= randn(rng, eltype(weights), size(weights)) .* std
    return weights
end

function ipa_point_weights_init!(weights)
    weights .= eltype(weights)(0.541324854612918)
    return weights
end

function torch_linear_init!(weights, bias=nothing)
    fan_in = size(weights, 2)
    bound = inv(sqrt(fan_in))
    rng = Random.default_rng()
    weights .= rand(rng, eltype(weights), size(weights)) .* (2 * bound) .- bound
    if bias !== nothing
        bias .= rand(rng, eltype(bias), length(bias)) .* (2 * bound) .- bound
    end
    return weights
end
