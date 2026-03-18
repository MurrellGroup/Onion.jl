# Fused residual addition + RMSNorm.
# Computes: new_x = residual + x, normed = rmsnorm(new_x, w)
# Returns (new_x, normed) — avoids materializing the intermediate.

function fused_add_rms_norm(b::Backend,
    residual::AbstractArray{T}, x::AbstractArray{T},
    w::AbstractVector;
    eps, offset,
) where T
    _fused_add_rms_norm(b, residual, x, w; eps, offset)
end

function _fused_add_rms_norm(b::DefaultBackend,
    residual::AbstractArray{T}, x::AbstractArray{T},
    w::AbstractVector;
    eps, offset,
) where T
    new_x = residual .+ x
    normed = _rms_norm(b, reshape(new_x, Keep(), :), w, Val(1); eps, offset)
    normed = reshape(normed, size(new_x))
    return new_x, normed
end
