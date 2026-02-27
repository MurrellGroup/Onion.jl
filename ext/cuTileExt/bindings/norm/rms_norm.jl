import ChainRulesCore as CRC

function Onion.rms_norm(::cuTileBackend,
    x::AbstractMatrix, w::AbstractVector;
    eps, offset
)
    y, _ = rms_norm(x, w; eps, offset)
    return y
end

function CRC.rrule(
    ::typeof(Onion.rms_norm), ::cuTileBackend,
    x::AbstractMatrix, w::AbstractVector; eps, offset
)
    y, rstd = rms_norm(x, w; eps, offset)
    function rms_norm_pullback(ȳ)
        dx, dw = ∇rms_norm(unthunk(ȳ), x, w, rstd; offset)
        return NoTangent(), NoTangent(), dx, dw
    end
    return y, rms_norm_pullback
end
