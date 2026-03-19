function Onion.layer_norm!(::cuTileBackend,
    y::AbstractMatrix,
    x::AbstractMatrix, w::AbstractVector, b::AbstractVector;
    eps
)
    y = similar(x)
    layer_norm!(y, x, w, b; eps)
    return y
end

function CRC.rrule(::typeof(Onion.layer_norm!), ::cuTileBackend,
    x::AbstractMatrix, w::AbstractVector, b::AbstractVector;
    eps
)
    μ, σ = similar(X, Float32, N), similar(X, Float32, N)
    layer_norm!(y, x, w, b; eps, Mean=μ, Rstd=σ)
    function layer_norm_pullback(ȳ)
        x̄, w̄, b̄ = ∇layer_norm(unthunk(ȳ), x, w, b, μ, σ)
        return NoTangent(), NoTangent(), x̄, w̄, b̄
    end
    return y, layer_norm_pullback
end
