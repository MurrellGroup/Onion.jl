# Fused QK-norm + partial RoPE for attention decode.

"""
    fused_qknorm_rope(q, k, q_norm_w, k_norm_w, cos, sin;
                      eps, offset, rotary_dim) -> (q_out, k_out)

Fused QK-norm + partial RoPE. Normalizes Q/K per-head, then applies
rotary embeddings to the first `rotary_dim` dimensions.
"""
@primitive fused_qknorm_rope
@primitive fused_qknorm_rope!

get_buffers(::typeof(fused_qknorm_rope), b::Backend, q, k, args...; kws...) =
    (; q_out = similar(q), k_out = similar(k))

function fused_qknorm_rope(::DefaultBackend,
    q, k, q_norm_weight, k_norm_weight, rope_cos, rope_sin;
    eps, offset, rotary_dim,
)
    # Norm (reshape to 2D for rms_norm, then back)
    q_n = rms_norm(DefaultBackend(), reshape(q, Keep(), :), q_norm_weight; eps, offset)
    k_n = rms_norm(DefaultBackend(), reshape(k, Keep(), :), k_norm_weight; eps, offset)
    q_n = reshape(q_n, size(q))
    k_n = reshape(k_n, size(k))

    # Partial RoPE (views to avoid copies)
    rd = rotary_dim
    rest = ntuple(_ -> Colon(), ndims(q) - 1)
    q_rot = @view q_n[1:rd, rest...]
    q_pass = @view q_n[rd+1:end, rest...]
    k_rot = @view k_n[1:rd, rest...]
    k_pass = @view k_n[rd+1:end, rest...]

    q_rotated = rotary_pos_emb(DefaultBackend(), q_rot, rope_cos, rope_sin)
    k_rotated = rotary_pos_emb(DefaultBackend(), k_rot, rope_cos, rope_sin)

    return vcat(q_rotated, q_pass), vcat(k_rotated, k_pass)
end
