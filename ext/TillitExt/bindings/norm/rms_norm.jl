function Onion.rms_norm!(::TillitBackend,
    y::AbstractMatrix,
    x::AbstractMatrix, w::AbstractVector;
    eps, offset
)
    Tillit.rms_norm!(y, x, w; eps, offset)
    return y
end

function CRC.rrule(
    ::typeof(Onion.rms_norm!), backend::TillitBackend,
    y::AbstractMatrix,
    x::AbstractMatrix, w::AbstractVector;
    eps, offset
)
    rstd = similar(x, Float32, size(x, 2))
    Onion.rms_norm!(backend, y, x, w; Rstd=rstd, eps, offset)
    function rms_norm_pullback(ȳ)
        x̄, w̄ = Tillit.∇rms_norm(unthunk(ȳ), x, w, rstd; offset)
        return NoTangent(), NoTangent(), NoTangent(), x̄, w̄
    end
    return y, rms_norm_pullback
end
