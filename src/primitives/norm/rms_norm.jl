using Statistics: mean
using Base.Broadcast: materialize

_rms_norm(x, eps) = @lazy x / √($mean(abs2, x, dims=1) + eps)
_rms_norm(x, w, eps, offset) = @lazy (w .+ offset) .* $_rms_norm(x, eps)

@impl DefaultBackend function rms_norm(
    x::AbstractArray, w::AbstractVector; eps, offset
)
    return materialize(_rms_norm(x, w, eps, offset))
end

@impl DefaultBackend function rms_norm(
    x::AbstractArray; eps
)
    return materialize(_rms_norm(x, eps))
end
