function Onion.layer_norm(::cuTileBackend,
    x::AbstractMatrix, w::AbstractVector, b::AbstractVector;
    eps
)
    y, _, _ = layer_norm(x, w, b; eps)
    return y
end

function CRC.rrule(::typeof(Onion.layer_norm), ::cuTileBackend,
    x::AbstractMatrix, w::AbstractVector, b::AbstractVector;
    eps
)
    y, μ, σ = layer_norm(x, w, b; eps)
    function layer_norm_pullback(ȳ)
        x̄, w̄, b̄ = ∇layer_norm(CRC.unthunk(ȳ), x, w, b, μ, σ)
        return CRC.NoTangent(), CRC.NoTangent(), x̄, w̄, b̄
    end
    return y, layer_norm_pullback
end
