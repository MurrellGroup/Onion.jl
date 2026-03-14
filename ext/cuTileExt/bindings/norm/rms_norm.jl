import ChainRulesCore as CRC

function Onion._rms_norm(::cuTileBackend,
    x::AbstractMatrix, w::AbstractVector, ::Val{1};
    eps, offset
)
    y, _ = rms_norm(x, w; eps, offset)
    return y
end

function CRC.rrule(
    ::typeof(Onion._rms_norm), ::cuTileBackend,
    x::AbstractMatrix, w::AbstractVector, ::Val{1};
    eps, offset
)
    y, rstd = rms_norm(x, w; eps, offset)
    function rms_norm_pullback(ȳ)
        dx, dw = ∇rms_norm(unthunk(ȳ), x, w, rstd; offset)
        return NoTangent(), NoTangent(), dx, dw, NoTangent()
    end
    return y, rms_norm_pullback
end
