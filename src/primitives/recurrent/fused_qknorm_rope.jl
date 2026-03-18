# Fused QK-norm + partial RoPE for attention decode.
# For each head: RMSNorm(q/k), then apply rotary embeddings to first rotary_dim dims.
# Returns (q_normed_roped, k_normed_roped) ready for attention.

function fused_qknorm_rope(b::Backend,
    q::AbstractArray{T}, k::AbstractArray{T},       # (head_dim, 1, num_heads, ...)
    q_norm_weight::AbstractVector, k_norm_weight::AbstractVector,
    rope_cos::AbstractArray, rope_sin::AbstractArray; # (rotary_dim/2, ...) for this position
    eps, offset, rotary_dim::Int,
) where T
    _fused_qknorm_rope(b, q, k, q_norm_weight, k_norm_weight, rope_cos, rope_sin;
        eps, offset, rotary_dim)
end

function _fused_qknorm_rope(::DefaultBackend,
    q, k, q_norm_weight, k_norm_weight, rope_cos, rope_sin;
    eps, offset, rotary_dim,
)
    # Norm
    q_n = _rms_norm(DefaultBackend(), reshape(q, Keep(), :), q_norm_weight, Val(1); eps, offset)
    k_n = _rms_norm(DefaultBackend(), reshape(k, Keep(), :), k_norm_weight, Val(1); eps, offset)
    q_n = reshape(q_n, size(q))
    k_n = reshape(k_n, size(k))

    # Partial RoPE (views to avoid copies)
    rd = rotary_dim
    rest = ntuple(_ -> Colon(), ndims(q) - 1)
    q_rot = @view q_n[1:rd, rest...]
    q_pass = @view q_n[rd+1:end, rest...]
    k_rot = @view k_n[1:rd, rest...]
    k_pass = @view k_n[rd+1:end, rest...]

    q_rotated = _rotary_pos_emb(DefaultBackend(), q_rot, rope_cos, rope_sin)
    k_rotated = _rotary_pos_emb(DefaultBackend(), k_rot, rope_cos, rope_sin)

    return vcat(q_rotated, q_pass), vcat(k_rotated, k_pass)
end
