function Onion._layer_norm(::cuTileBackend,
    x::AbstractMatrix, w::AbstractVector, b::AbstractVector, ::Val{1};
    eps
)
    y, _, _ = layer_norm(x, w, b; eps)
    return y
end

function CRC.rrule(::typeof(Onion._layer_norm), ::cuTileBackend,
    x::AbstractMatrix, w::AbstractVector, b::AbstractVector, ::Val{1};
    eps
)
    y, μ, σ = layer_norm(x, w, b; eps)
    function layer_norm_pullback(ȳ)
        x̄, w̄, b̄ = ∇layer_norm(unthunk(ȳ), x, w, b, μ, σ)
        return NoTangent(), NoTangent(), x̄, w̄, b̄, NoTangent()
    end
    return y, layer_norm_pullback
end
