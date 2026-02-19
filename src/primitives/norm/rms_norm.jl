using Statistics: mean
using Base.Broadcast: materialize

_rms_norm(x, eps) = @lazy x / √($mean(abs2, x, dims=1) + eps)
_rms_norm(x, w, eps, offset) = @lazy (w .+ offset) .* $_rms_norm(x, eps)

function rms_norm(::DefaultBackend,
    x::AbstractArray, w::AbstractVector; eps, offset
)
    return materialize(_rms_norm(x, w, eps, offset))
end

function rms_norm(::DefaultBackend,
    x::AbstractArray; eps
)
    return materialize(_rms_norm(x, eps))
end
