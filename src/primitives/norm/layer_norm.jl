using Statistics: mean, var

function layer_norm(::DefaultBackend,
    x::AbstractArray,
    w::AbstractVector, b::AbstractVector,
    dims::Int = 1;
    eps
)
    μ = mean(x; dims)
    σ² = var(x; dims, mean=μ, corrected=false)
    (x .- μ) ./ sqrt.(σ² .+ eps) .* w .+ b
end
