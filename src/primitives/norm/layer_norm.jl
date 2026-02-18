using Statistics: mean, var

@impl DefaultBackend function layer_norm(x::AbstractArray, w::AbstractVector, b::AbstractVector; eps)
    μ = mean(x; dims=1)
    σ² = var(x; dims=1, mean=μ, corrected=false)
    (x .- μ) ./ sqrt.(σ² .+ eps) .* w .+ b
end
