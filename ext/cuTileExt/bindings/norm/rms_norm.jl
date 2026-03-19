import ChainRulesCore as CRC

function Onion.rms_norm!(::cuTileBackend,
    y::AbstractMatrix,
    x::AbstractMatrix, w::AbstractVector;
    eps, offset
)
    rms_norm!(y, x, w; eps, offset)
    return y
end

function CRC.rrule(
    ::typeof(Onion.rms_norm!), ::cuTileBackend,
    y::AbstractMatrix,
    x::AbstractMatrix, w::AbstractVector;
    eps, offset
)
    rstd = similar(x, Float32, size(x, 2))
    rms_norm!(y, x, w; Rstd=rstd, eps, offset)
    function rms_norm_pullback(ȳ)
        x̄, w̄ = ∇rms_norm(unthunk(ȳ), x, w, rstd; offset)
        return NoTangent(), NoTangent(), x̄, w̄
    end
    return y, rms_norm_pullback
end
