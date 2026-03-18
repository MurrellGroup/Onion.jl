# Fused residual addition + RMSNorm.
# Computes: new_x = residual + x, normed = rmsnorm(new_x, w)
# Returns (new_x, normed) — avoids materializing the intermediate.

"""
    fused_add_rms_norm(residual, x, w; eps, offset) -> (new_x, normed)

Fused residual add + RMSNorm: `new_x = residual + x`, `normed = rmsnorm(new_x, w)`.
Returns both the raw sum (next residual) and the normalized output.
Handles reshaping to 2D internally (this is a fused operation).
"""
@primitive fused_add_rms_norm
@primitive fused_add_rms_norm!

function fused_add_rms_norm!(b::DefaultBackend,
    new_x::AbstractArray{T}, normed::AbstractArray{T},
    residual::AbstractArray{T}, x::AbstractArray{T},
    w::AbstractVector;
    eps, offset,
) where T
    new_x .= residual .+ x
    rms_norm!(b, reshape(normed, Keep(), :), reshape(new_x, Keep(), :), w; eps, offset)
    return new_x, normed
end

function fused_add_rms_norm(b::DefaultBackend,
    residual::AbstractArray{T}, x::AbstractArray{T},
    w::AbstractVector;
    eps, offset,
) where T
    new_x = residual .+ x
    normed = rms_norm(b, reshape(new_x, Keep(), :), w; eps, offset)
    normed = reshape(normed, size(new_x))
    return new_x, normed
end
