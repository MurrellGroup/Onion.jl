# AnyGPUArray layer overrides for performance-critical protein layers.
# Uses in-place operations (.+=, .=) that work on any GPU backend.
# No buffer pool, no CUDA.unsafe_free! — those are OnionTile-specific.
#
# AD compatibility: uses ONIONop.within_gradient to select in-place (inference)
# vs out-of-place (training) operations. within_gradient(x) returns false
# normally and true inside Zygote pullbacks via its rrule.

using GPUArraysCore: AnyGPUArray
using ONIONop: within_gradient

# ============================================================================
# Flash IPA: augmented flash attention + batched_mul pair aggregation
#
# Shared core for ESMFoldIPA, InvariantPointAttention, MultimerInvariantPointAttention.
#
# Reformulates IPA as:
#   1. Augmented Q/K/V → flash_attention_bias_forward (cuTile/ONIONop kernel)
#   2. Pair aggregation → batched_mul loop over heads (no 5D broadcast)
#
# Mathematical basis:
#   ||T_i q_p - T_j k_p||^2 = ||T_i q_p||^2 + ||T_j k_p||^2 - 2(T_i q_p)·(T_j k_p)
#   Query norms cancel in softmax; key norms become additive bias; cross-terms
#   become augmented Q·K dot products. Pair aggregation can't be fused (z depends
#   on both i,j) but batched_mul avoids the huge (C_z, L, L, H, B) broadcast.
# ============================================================================

function _flash_ipa_core(
    q, k, v, q_pts, k_pts, v_pts,
    linear_b, head_weights_raw, linear_out,
    s, z, r, mask,
    c_h, H, Pq, Pv, C_z, eps, inf_val,
)
    L = size(s, 2); B = size(s, 3); T = eltype(s)
    D_aug = c_h + 3 * max(Pq, Pv)
    D_pad = nextpow(2, D_aug)

    # ─── Scaling factors ───
    α = T((3 * c_h)^(-0.25))
    hw_raw = NNlib.softplus.(head_weights_raw)
    hw = hw_raw .* T(sqrt(1 / (3 * (Pq * 9 / 2))))
    γ = sqrt.(hw)  # (H,)

    # ─── Augmented Q, K, V ───
    qpf = reshape(q_pts, 3*Pq, H, L, B) .* reshape(γ, 1, H, 1, 1)
    kpf = reshape(k_pts, 3*Pq, H, L, B) .* reshape(γ, 1, H, 1, 1)

    Q_aug = cat(q .* α, qpf; dims=1)
    K_aug = cat(k .* α, kpf; dims=1)
    V_aug = cat(v, v_pts[1,:,:,:,:], v_pts[2,:,:,:,:], v_pts[3,:,:,:,:]; dims=1)

    # Pad to power-of-2 for cuTile/ONIONop tile alignment
    total_pad = D_pad - size(Q_aug, 1)
    if total_pad > 0
        zp = fill!(similar(s, T, total_pad, H, L, B), zero(T))
        Q_aug = cat(Q_aug, zp; dims=1)
        K_aug = cat(K_aug, fill!(similar(zp), zero(T)); dims=1)
    end
    v_pad = D_pad - size(V_aug, 1)
    if v_pad > 0
        V_aug = cat(V_aug, fill!(similar(s, T, v_pad, H, L, B), zero(T)); dims=1)
    end

    # Permute to flash attention layout: (D, L, H, B)
    Qf = permutedims(Q_aug, (1, 3, 2, 4))
    Kf = permutedims(K_aug, (1, 3, 2, 4))
    Vf = permutedims(V_aug, (1, 3, 2, 4))

    # ─── Effective bias: (L_k, L_q, H, B) ───
    pb = linear_b(z)  # (H, Lq, Lk, B)
    kn = dropdims(sum(k_pts .^ 2; dims=(1,2)); dims=(1,2))  # (H, L, B)
    knb = T(-0.5) .* reshape(hw, H, 1, 1) .* kn  # (H, Lk, B)
    sq = reshape(mask, L, 1, B) .* reshape(mask, 1, L, B)  # (Lq, Lk, B)
    mb = inf_val .* (sq .- one(T))  # (Lq, Lk, B)

    bias_eff = T(sqrt(1/3)) .* permutedims(pb, (3,2,1,4)) .+
               reshape(permutedims(knb, (2,1,3)), L, 1, H, B) .+
               reshape(permutedims(mb, (2,1,3)), L, L, 1, B)

    # ─── Flash attention: softmax(Q·K^T + bias) · V ───
    O = flash_attention_bias_forward(Qf, Kf, Vf, bias_eff; scale=one(T))

    # ─── Extract scalar output → (H*C, L, B) ───
    os = permutedims(O[1:c_h, :, :, :], (4,2,3,1))
    os = permutedims(os, (1,2,4,3))
    o_feat = permutedims(reshape(os, B, L, H*c_h), (3,2,1))

    # ─── Extract point outputs → inverse transform → norms ───
    opx = permutedims(O[c_h+1:c_h+Pv, :, :, :], (4,2,3,1))
    opy = permutedims(O[c_h+Pv+1:c_h+2Pv, :, :, :], (4,2,3,1))
    opz = permutedims(O[c_h+2Pv+1:c_h+3Pv, :, :, :], (4,2,3,1))

    opt = cat(opx, opy, opz; dims=5)
    opt = invert_apply_rigid(r, permutedims(opt, (5,4,3,2,1)))
    opt = permutedims(opt, (5,4,3,2,1))

    optn = sqrt.(sum(opt .^ 2; dims=5) .+ eps)
    optn = reshape(permutedims(dropdims(optn; dims=5), (1,2,4,3)), B, L, H*Pv)
    o_pt_norm = permutedims(optn, (3,2,1))

    opt2 = reshape(permutedims(opt, (1,2,4,3,5)), B, L, H*Pv, 3)
    o_px = permutedims(opt2[:,:,:,1], (3,2,1))
    o_py = permutedims(opt2[:,:,:,2], (3,2,1))
    o_pz = permutedims(opt2[:,:,:,3], (3,2,1))

    # ─── Pair aggregation: recompute attention + batched_mul ───
    Qf3 = reshape(Qf, D_pad, L, H*B)
    Kf3 = reshape(Kf, D_pad, L, H*B)
    logits = NNlib.batched_mul(permutedims(Qf3, (2,1,3)), Kf3)
    bias3 = reshape(bias_eff, L, L, H*B)
    logits = logits .+ permutedims(bias3, (2,1,3))
    logits_max = maximum(logits; dims=2)
    logits_exp = exp.(logits .- logits_max)
    attn = logits_exp ./ sum(logits_exp; dims=2)

    # batched_mul pair aggregation (loop over heads, no 5D broadcast)
    attn4 = reshape(attn, L, L, H, B)
    z_kq = permutedims(z, (1, 3, 2, 4))       # (C_z, L_k, L_q, B)
    z_3d = reshape(z_kq, C_z, L, L * B)       # (C_z, L_k, L_q*B)
    pair_out = similar(z, T, C_z, L, H, B)
    for h in 1:H
        p_h = permutedims(attn4[:, :, h, :], (2, 1, 3))  # (L_k, L_q, B)
        p_h3 = reshape(p_h, L, 1, L * B)
        pair_h = NNlib.batched_mul(z_3d, p_h3)
        pair_out[:, :, h, :] .= reshape(pair_h, C_z, L, B)
    end
    o_pair = reshape(permutedims(pair_out, (1,3,2,4)), H*C_z, L, B)

    # ─── Concat + output projection ───
    concat = cat(o_feat, o_px, o_py, o_pz, o_pt_norm, o_pair; dims=1)
    return linear_out(concat)
end

# ─── ESMFoldIPA / InvariantPointAttention (fused kv) ───
function (m::ESMFoldIPA)(s::AnyGPUArray, z::AnyGPUArray, r, mask::AnyGPUArray)
    c_h = m.c_hidden; H = m.no_heads
    Pq = m.no_qk_points; Pv = m.no_v_points

    q = reshape(m.linear_q(s), c_h, H, size(s, 2), size(s, 3))
    kv = reshape(m.linear_kv(s), 2*c_h, H, size(s, 2), size(s, 3))
    k = kv[1:c_h, :, :, :]
    v = kv[c_h+1:end, :, :, :]

    q_pts = m.linear_q_points(s, r)     # (3, Pq, H, L, B)
    kv_pts = m.linear_kv_points(s, r)   # (3, Pq+Pv, H, L, B)
    k_pts = kv_pts[:, 1:Pq, :, :, :]
    v_pts = kv_pts[:, Pq+1:end, :, :, :]

    return _flash_ipa_core(
        q, k, v, q_pts, k_pts, v_pts,
        m.linear_b, m.head_weights, m.linear_out,
        s, z, r, mask,
        c_h, H, Pq, Pv, m.c_z, m.eps, m.inf,
    )
end

# ─── MultimerInvariantPointAttention (separate q/k/v) ───
function (m::MultimerInvariantPointAttention)(s::AnyGPUArray, z::AnyGPUArray, r, mask::AnyGPUArray)
    c_h = m.c_hidden; H = m.no_heads

    q = reshape(m.linear_q(s), c_h, H, size(s, 2), size(s, 3))
    k = reshape(m.linear_k(s), c_h, H, size(s, 2), size(s, 3))
    v = reshape(m.linear_v(s), c_h, H, size(s, 2), size(s, 3))

    q_pts = m.linear_q_points(s, r)   # (3, Pq, H, L, B)
    k_pts = m.linear_k_points(s, r)   # (3, Pq, H, L, B)
    v_pts = m.linear_v_points(s, r)   # (3, Pv, H, L, B)

    return _flash_ipa_core(
        q, k, v, q_pts, k_pts, v_pts,
        m.linear_b, m.head_weights, m.linear_out,
        s, z, r, mask,
        c_h, H, m.no_qk_points, m.no_v_points, m.c_z, m.eps, m.inf,
    )
end

# ============================================================================
# TriangularSelfAttentionBlock: in-place residual additions
# This is the most important override — 7 residual additions that each
# allocate a new array in the CPU path.
# ============================================================================

function (m::TriangularSelfAttentionBlock)(sequence_state::AnyGPUArray, pairwise_state::AnyGPUArray;
        mask=nothing, chunk_size=nothing, residue_index=nothing)
    bias = m.pair_to_sequence(pairwise_state)
    y = m.layernorm_1(sequence_state)
    y, _ = m.seq_attention(y; mask=mask, bias=bias)

    if within_gradient(sequence_state)
        sequence_state = sequence_state .+ m.drop(y)
    else
        sequence_state .+= m.drop(y)
    end
    sequence_state = m.mlp_seq(sequence_state)

    if within_gradient(pairwise_state)
        pairwise_state = pairwise_state .+ m.sequence_to_pair(sequence_state)
    else
        pairwise_state .+= m.sequence_to_pair(sequence_state)
    end

    tri_mask = mask === nothing ? nothing :
        (reshape(mask, size(mask, 1), 1, size(mask, 2)) .* reshape(mask, 1, size(mask, 1), size(mask, 2)))

    if within_gradient(pairwise_state)
        pairwise_state = pairwise_state .+ m.row_drop(m.tri_mul_out(pairwise_state; mask=tri_mask))
        pairwise_state = pairwise_state .+ m.col_drop(m.tri_mul_in(pairwise_state; mask=tri_mask))
        pairwise_state = pairwise_state .+ m.row_drop(m.tri_att_start(pairwise_state; mask=tri_mask))
        pairwise_state = pairwise_state .+ m.col_drop(m.tri_att_end(pairwise_state; mask=tri_mask))
    else
        pairwise_state .+= m.row_drop(m.tri_mul_out(pairwise_state; mask=tri_mask))
        pairwise_state .+= m.col_drop(m.tri_mul_in(pairwise_state; mask=tri_mask))
        pairwise_state .+= m.row_drop(m.tri_att_start(pairwise_state; mask=tri_mask))
        pairwise_state .+= m.col_drop(m.tri_att_end(pairwise_state; mask=tri_mask))
    end

    pairwise_state = m.mlp_pair(pairwise_state)

    return sequence_state, pairwise_state
end

# ============================================================================
# ResidueMLP: in-place ReLU + in-place residual
# ============================================================================

function (m::ResidueMLP)(x::AnyGPUArray)
    y = m.norm(x)
    y = m.fc1(y)
    if within_gradient(y)
        y = max.(y, 0f0)  # out-of-place ReLU
    else
        y .= max.(y, 0f0)  # in-place ReLU
    end
    y = m.fc2(y)
    y = m.dropout(y)
    if within_gradient(x)
        x = x .+ y  # out-of-place residual
    else
        x .+= y  # in-place residual
    end
    return x
end

# ============================================================================
# StructureModuleTransitionLayer: in-place ReLU + in-place residual
# ============================================================================

function (m::StructureModuleTransitionLayer)(s::AnyGPUArray)
    s0 = s
    s = m.linear_1(s)
    if within_gradient(s)
        s = max.(s, 0f0)
        s = m.linear_2(s)
        s = max.(s, 0f0)
    else
        s .= max.(s, 0f0)
        s = m.linear_2(s)
        s .= max.(s, 0f0)
    end
    s = m.linear_3(s)
    if within_gradient(s)
        s = s .+ s0  # out-of-place residual
    else
        s .+= s0  # in-place residual
    end
    return s
end

# ============================================================================
# AngleResnetBlock: in-place residual
# ============================================================================

function (m::AngleResnetBlock)(a::AnyGPUArray)
    s = a
    a = m.linear_1(max.(a, 0f0))
    a = m.linear_2(max.(a, 0f0))
    if within_gradient(a)
        a = a .+ s  # out-of-place residual
    else
        a .+= s  # in-place residual
    end
    return a
end

# ============================================================================
# SequenceToPair: pre-allocated outer product buffer via similar()
# Avoids separate prod + diff + cat allocations.
# In gradient mode, uses cat() instead of view-writes for AD compatibility.
# ============================================================================

function (m::SequenceToPair)(sequence_state::AnyGPUArray)
    s_norm = m.layernorm(sequence_state)
    s = m.proj(s_norm)
    inner_dim = size(s, 1) ÷ 2
    q = view(s, 1:inner_dim, :, :)
    k = view(s, (inner_dim + 1):(2 * inner_dim), :, :)
    L = size(q, 2)
    B = size(q, 3)
    q_exp = reshape(q, inner_dim, 1, L, B)
    k_exp = reshape(k, inner_dim, L, 1, B)

    if within_gradient(s)
        # Out-of-place: AD-compatible
        prod = q_exp .* k_exp
        diff = q_exp .- k_exp
        x = cat(prod, diff; dims=1)
    else
        # Pre-allocated buffer for product and difference
        x = similar(s, 2 * inner_dim, L, L, B)
        prod_view = @view x[1:inner_dim, :, :, :]
        diff_view = @view x[inner_dim+1:end, :, :, :]
        @. prod_view = q_exp * k_exp
        @. diff_view = q_exp - k_exp
    end

    x = m.o_proj(x)
    return x
end
