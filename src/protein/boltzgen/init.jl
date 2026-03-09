using Random

const _BG_TRUNCNORM_STD = 0.8796256610342398

_bg_prod(nums) = foldl(*, nums; init=1)

function _bg_calculate_fan(weight_shape::NTuple{2,Int}, fan::Symbol)
    fan_out, fan_in = weight_shape
    if fan === :fan_in
        return fan_in
    elseif fan === :fan_out
        return fan_out
    elseif fan === :fan_avg
        return (fan_in + fan_out) / 2
    else
        error("Invalid fan option: $fan")
    end
end

function trunc_normal_init!(weights; scale=1.0, fan::Symbol=:fan_in)
    shape = size(weights)
    if length(shape) != 2
        error("trunc_normal_init! expects 2D weight, got size $(shape)")
    end
    f = _bg_calculate_fan(shape, fan)
    scale = scale / max(1, f)
    std = sqrt(scale) / _BG_TRUNCNORM_STD

    wflat = reshape(weights, :)
    rng = Random.default_rng()
    for i in eachindex(wflat)
        v = randn(rng)
        while v < -2 || v > 2
            v = randn(rng)
        end
        wflat[i] = v * std
    end
    return weights
end

lecun_normal_init!(weights) = trunc_normal_init!(weights; scale=1.0)
he_normal_init!(weights) = trunc_normal_init!(weights; scale=2.0)

function glorot_uniform_init!(weights)
    shape = size(weights)
    if length(shape) != 2
        error("glorot_uniform_init! expects 2D weight, got size $(shape)")
    end
    fan_out, fan_in = shape
    limit = sqrt(6 / (fan_in + fan_out))
    rng = Random.default_rng()
    weights .= (rand(rng, eltype(weights), size(weights)) .* (2 * limit) .- limit)
    return weights
end

function final_init!(weights)
    weights .= zero(eltype(weights))
    return weights
end

gating_init!(weights) = final_init!(weights)

function bias_init_zero!(bias)
    bias .= zero(eltype(bias))
    return bias
end

function bias_init_one!(bias)
    bias .= one(eltype(bias))
    return bias
end

function normal_init!(weights)
    shape = size(weights)
    if length(shape) != 2
        error("normal_init! expects 2D weight, got size $(shape)")
    end
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
