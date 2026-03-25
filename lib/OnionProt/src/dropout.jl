using Random

# ──── Training mode toggle ────

const _TRAINING = Ref(false)

training_mode() = _TRAINING[]
training_mode!(flag::Bool) = (_TRAINING[] = flag)

# ──── SharedDropout ────

@concrete struct SharedDropout <: Layer
    rate::Float32
    broadcast_dims  # Tuple of dims set to size 1 in the mask
end

SharedDropout(rate::Real, dims::Int) = SharedDropout(Float32(rate), (dims,))
SharedDropout(rate::Real, dims) = SharedDropout(Float32(rate), Tuple(dims))

function (m::SharedDropout)(x)
    (m.rate == 0f0 || !_TRAINING[]) && return x
    T = eltype(x)
    shape = ntuple(ndims(x)) do d
        d in m.broadcast_dims ? 1 : size(x, d)
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
