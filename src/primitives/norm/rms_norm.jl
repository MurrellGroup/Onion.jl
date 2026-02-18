using Statistics: mean

@impl DefaultBackend function rms_norm(x::AbstractArray, w::AbstractVector; eps, offset)
    y = (w .+ offset) .* x ./ .√(mean(abs2, x, dims=1) .+ eps)
    return y
end

@impl DefaultBackend function rms_norm(x::AbstractArray; eps)
    y = x ./ .√(mean(abs2, x, dims=1) .+ eps)
    return y
end
