using Flux

function get_dropout_mask(
    dropout::Real,
    z::AbstractArray,
    training::Bool=bg_istraining();
    columnwise::Bool=false,
)
    eff_dropout = eltype(z)(dropout) * (training ? one(eltype(z)) : zero(eltype(z)))
    # When not training (or dropout == 0), mask is all 1 — return scalar for GPU compat
    if eff_dropout ≈ 0
        return one(eltype(z))
    end
    if columnwise
        v = z[1:1, 1:1, :, :]
    else
        v = z[1:1, :, 1:1, :]
    end
    d = rand!(similar(z, Float32, size(v))) .> eff_dropout
    d = eltype(z).(d)
    scale = eltype(z)(1 / (1 - eff_dropout))
    d = d .* scale
    return d
end

function get_dropout_mask_columnwise(
    dropout::Real,
    z::AbstractArray,
    training::Bool=bg_istraining(),
)
    return get_dropout_mask(dropout, z, training; columnwise=true)
end

function get_dropout_mask_rowise(
    dropout::Real,
    z::AbstractArray,
    training::Bool=bg_istraining(),
)
    return get_dropout_mask(dropout, z, training; columnwise=false)
end
