using NNlib
using Statistics
import Zygote

struct StructureModuleConfig
    c_s::Int
    c_z::Int
    c_ipa::Int
    c_resnet::Int
    no_heads_ipa::Int
    no_qk_points::Int
    no_v_points::Int
    dropout_rate::Float32
    no_blocks::Int
    no_transition_layers::Int
    no_resnet_blocks::Int
    no_angles::Int
    trans_scale_factor::Float32
    epsilon::Float32
    inf::Float32
end

function StructureModuleConfig()
    return StructureModuleConfig(
        384, 128, 16, 128, 12, 4, 8,
        0.1f0, 8, 1, 2, 7, 10f0,
        Float32(1e-8), Float32(1e5),
    )
end

@concrete struct PointProjection <: Onion.Layer
    linear
    num_points::Int
    no_heads::Int
end

@layer PointProjection

function PointProjection(c_hidden::Int, num_points::Int, no_heads::Int)
    linear = LinearFirst(c_hidden, no_heads * 3 * num_points)
    return PointProjection(linear, num_points, no_heads)
end

function (m::PointProjection)(activations, rigids)
    raw = m.linear(activations)
    H = m.no_heads
    P = m.num_points
    raw = reshape(raw, P, H, 3, size(raw, 2), size(raw, 3))
    points_local = permutedims(raw, (3, 1, 2, 4, 5))
    points_global = apply_rigid(rigids, points_local)
    return points_global
end

@concrete struct ESMFoldIPA <: Onion.Layer
    c_s::Int
    c_z::Int
    c_hidden::Int
    no_heads::Int
    no_qk_points::Int
    no_v_points::Int
    linear_q
    linear_q_points
    linear_kv
    linear_kv_points
    linear_b
    head_weights
    linear_out
    eps::Float32
    inf::Float32
end

@layer ESMFoldIPA

function ESMFoldIPA(c_s::Int, c_z::Int, c_hidden::Int, no_heads::Int, no_qk_points::Int, no_v_points::Int; inf::Real=1e5, eps::Real=1e-8)
    linear_q = LinearFirst(c_s, c_hidden * no_heads)
    linear_q_points = PointProjection(c_s, no_qk_points, no_heads)
    linear_kv = LinearFirst(c_s, 2 * c_hidden * no_heads)
    linear_kv_points = PointProjection(c_s, no_qk_points + no_v_points, no_heads)
    linear_b = LinearFirst(c_z, no_heads)
    head_weights = fill(0.54132485f0, no_heads)
    linear_out = LinearFirst(no_heads * (c_z + c_hidden + no_v_points * 4), c_s)
    return ESMFoldIPA(
        c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points,
        linear_q, linear_q_points, linear_kv, linear_kv_points,
        linear_b, head_weights, linear_out, Float32(eps), Float32(inf),
    )
end

function (m::ESMFoldIPA)(s, z, r, mask)
    q = m.linear_q(s)
    q = reshape(q, m.c_hidden, m.no_heads, size(q, 2), size(q, 3))
    kv = m.linear_kv(s)
    kv = reshape(kv, 2 * m.c_hidden, m.no_heads, size(kv, 2), size(kv, 3))
    k = view(kv, 1:m.c_hidden, :, :, :)
    v = view(kv, (m.c_hidden + 1):(2 * m.c_hidden), :, :, :)

    q_bhlc = permutedims(q, (4, 2, 3, 1))
    k_bhlc = permutedims(k, (4, 2, 3, 1))
    k_bhcl = permutedims(k_bhlc, (1, 2, 4, 3))

    B = size(q_bhlc, 1)
    H = size(q_bhlc, 2)
    L = size(q_bhlc, 3)
    C = size(q_bhlc, 4)
    q3 = permutedims(reshape(q_bhlc, B * H, L, C), (2, 3, 1))
    k3 = permutedims(reshape(k_bhcl, B * H, C, L), (2, 3, 1))
    a3 = NNlib.batched_mul(q3, k3)
    a = reshape(a3, L, L, B, H)
    a = permutedims(a, (3, 4, 1, 2))

    a = a .* sqrt(1f0 / (3f0 * m.c_hidden))

    b = m.linear_b(z)
    b_perm = permutedims(b, (4, 1, 2, 3))
    a = a .+ sqrt(1f0 / 3f0) .* b_perm

    q_pts = m.linear_q_points(s, r)
    kv_pts = m.linear_kv_points(s, r)
    k_pts = view(kv_pts, :, 1:m.no_qk_points, :, :, :)
    v_pts = view(kv_pts, :, (m.no_qk_points + 1):(m.no_qk_points + m.no_v_points), :, :, :)

    q_pts = permutedims(q_pts, (5, 4, 3, 2, 1))
    k_pts = permutedims(k_pts, (5, 4, 3, 2, 1))
    v_pts = permutedims(v_pts, (5, 4, 3, 2, 1))

    q_exp = reshape(q_pts, size(q_pts, 1), size(q_pts, 2), 1, size(q_pts, 3), size(q_pts, 4), size(q_pts, 5))
    k_exp = reshape(k_pts, size(k_pts, 1), 1, size(k_pts, 2), size(k_pts, 3), size(k_pts, 4), size(k_pts, 5))
    pt_att = q_exp .- k_exp
    pt_att = sum(pt_att .^ 2; dims=6)

    head_weights = NNlib.softplus.(m.head_weights)
    head_weights = head_weights .* sqrt(1f0 / (3f0 * (m.no_qk_points * 9f0 / 2f0)))
    hw = reshape(head_weights, 1, 1, 1, m.no_heads, 1, 1)
    pt_att = sum(pt_att .* hw; dims=5) .* (-0.5f0)
    pt_att = dropdims(pt_att; dims=(5, 6))
    pt_att = permutedims(pt_att, (1, 4, 2, 3))

    square_mask = reshape(mask, size(mask, 1), 1, size(mask, 2)) .* reshape(mask, 1, size(mask, 1), size(mask, 2))
    square_mask = permutedims(square_mask, (3, 1, 2))
    square_mask = m.inf .* (square_mask .- 1)
    a = a .+ pt_att
    a = a .+ reshape(square_mask, size(square_mask, 1), 1, size(square_mask, 2), size(square_mask, 3))

    a = NNlib.softmax(a; dims=4)

    v_bhlc = permutedims(v, (4, 2, 3, 1))
    a3 = reshape(permutedims(a, (3, 4, 1, 2)), L, L, :)
    v3 = permutedims(reshape(v_bhlc, B * H, L, C), (2, 3, 1))
    o3 = NNlib.batched_mul(a3, v3)
    o = reshape(o3, L, C, B, H)
    o = permutedims(o, (3, 1, 4, 2))
    o = permutedims(o, (1, 2, 4, 3))
    o = reshape(o, B, L, H * C)

    v_pts_x = _view_last1(v_pts, 1)
    v_pts_y = _view_last1(v_pts, 2)
    v_pts_z = _view_last1(v_pts, 3)

    vpx = permutedims(v_pts_x, (2, 4, 1, 3))
    vpy = permutedims(v_pts_y, (2, 4, 1, 3))
    vpz = permutedims(v_pts_z, (2, 4, 1, 3))

    vpx3 = reshape(vpx, L, m.no_v_points, :)
    vpy3 = reshape(vpy, L, m.no_v_points, :)
    vpz3 = reshape(vpz, L, m.no_v_points, :)

    o_px = NNlib.batched_mul(a3, vpx3)
    o_py = NNlib.batched_mul(a3, vpy3)
    o_pz = NNlib.batched_mul(a3, vpz3)

    o_px = reshape(o_px, L, m.no_v_points, B, m.no_heads)
    o_py = reshape(o_py, L, m.no_v_points, B, m.no_heads)
    o_pz = reshape(o_pz, L, m.no_v_points, B, m.no_heads)

    o_px = permutedims(o_px, (3, 1, 4, 2))
    o_py = permutedims(o_py, (3, 1, 4, 2))
    o_pz = permutedims(o_pz, (3, 1, 4, 2))

    o_pt = cat(o_px, o_py, o_pz; dims=5)
    o_pt = invert_apply_rigid(r, permutedims(o_pt, (5, 4, 3, 2, 1)))
    o_pt = permutedims(o_pt, (5, 4, 3, 2, 1))

    o_pt_norm = sqrt.(sum(o_pt .^ 2; dims=5) .+ m.eps)
    o_pt_norm = dropdims(o_pt_norm; dims=5)
    o_pt_norm = permutedims(o_pt_norm, (1, 2, 4, 3))
    o_pt_norm = reshape(o_pt_norm, size(o_pt_norm, 1), size(o_pt_norm, 2), m.no_heads * m.no_v_points)

    o_pt = permutedims(o_pt, (1, 2, 4, 3, 5))
    o_pt = reshape(o_pt, size(o_pt)[1:end-3]..., m.no_heads * m.no_v_points, 3)
    o_px = _view_last1(o_pt, 1)
    o_py = _view_last1(o_pt, 2)
    o_pz = _view_last1(o_pt, 3)

    a_t = permutedims(a, (1, 2, 4, 3))
    a_exp = reshape(a_t, size(a_t, 1), size(a_t, 2), size(a_t, 3), size(a_t, 4), 1)
    z_swap = permutedims(z, (4, 3, 2, 1))
    z_exp = reshape(z_swap, size(z_swap, 1), 1, size(z_swap, 2), size(z_swap, 3), size(z_swap, 4))
    o_pair = sum(a_exp .* z_exp; dims=3)
    o_pair = dropdims(o_pair; dims=3)
    o_pair = permutedims(o_pair, (1, 3, 2, 4))
    o_pair = permutedims(o_pair, (1, 2, 4, 3))
    o_pair = reshape(o_pair, size(o_pair, 1), size(o_pair, 2), m.no_heads * m.c_z)

    o_feat = permutedims(o, (3, 2, 1))
    o_px = permutedims(o_px, (3, 2, 1))
    o_py = permutedims(o_py, (3, 2, 1))
    o_pz = permutedims(o_pz, (3, 2, 1))
    o_pt_norm = permutedims(o_pt_norm, (3, 2, 1))
    o_pair = permutedims(o_pair, (3, 2, 1))

    concat = cat(o_feat, o_px, o_py, o_pz, o_pt_norm, o_pair; dims=1)
    return m.linear_out(concat)
end

# InvariantPointAttention is structurally identical to ESMFoldIPA
# (same fields, same forward pass, same scaling).
const InvariantPointAttention = ESMFoldIPA

# ============================================================================
# PointProjectionMultimer: AF2 multimer point projection
# Same as PointProjection but with (3P, H) weight layout instead of (P, H, 3)
# ============================================================================

@concrete struct PointProjectionMultimer <: Onion.Layer
    linear
    num_points::Int
    no_heads::Int
end

@layer PointProjectionMultimer

function PointProjectionMultimer(c_hidden::Int, num_points::Int, no_heads::Int)
    linear = LinearFirst(c_hidden, no_heads * 3 * num_points)
    return PointProjectionMultimer(linear, num_points, no_heads)
end

function (m::PointProjectionMultimer)(activations::AbstractArray, rigids)
    raw = m.linear(activations) # (3*P*H, L, B) with (3P,H) row ordering
    P = m.num_points
    H = m.no_heads
    raw = reshape(raw, 3 * P, H, size(raw, 2), size(raw, 3)) # (3P, H, L, B)

    x = view(raw, 1:P, :, :, :)
    y = view(raw, (P + 1):(2 * P), :, :, :)
    z = view(raw, (2 * P + 1):(3 * P), :, :, :)

    points_local = cat(
        reshape(x, 1, P, H, size(raw, 3), size(raw, 4)),
        reshape(y, 1, P, H, size(raw, 3), size(raw, 4)),
        reshape(z, 1, P, H, size(raw, 3), size(raw, 4));
        dims=1,
    ) # (3, P, H, L, B)

    points_global = apply_rigid(rigids, points_local)
    return points_global
end

# ============================================================================
# MultimerInvariantPointAttention: AF2 multimer IPA
# Differs from ESMFoldIPA/InvariantPointAttention in:
#   - Separate linear_q, linear_k, linear_v (instead of fused linear_kv)
#   - Separate point projections (PointProjectionMultimer for each of q, k, v)
#   - sqrt(1/3) applied to all logits at end (instead of per-component)
# ============================================================================

@concrete struct MultimerInvariantPointAttention <: Onion.Layer
    c_s::Int
    c_z::Int
    c_hidden::Int
    no_heads::Int
    no_qk_points::Int
    no_v_points::Int
    linear_q
    linear_k
    linear_v
    linear_q_points
    linear_k_points
    linear_v_points
    linear_b
    head_weights
    linear_out
    eps::Float32
    inf::Float32
end

@layer MultimerInvariantPointAttention

function MultimerInvariantPointAttention(
    c_s::Int,
    c_z::Int,
    c_hidden::Int,
    no_heads::Int,
    no_qk_points::Int,
    no_v_points::Int;
    inf::Real=1e5,
    eps::Real=1e-8,
)
    linear_q = LinearFirst(c_s, c_hidden * no_heads; bias=false)
    linear_k = LinearFirst(c_s, c_hidden * no_heads; bias=false)
    linear_v = LinearFirst(c_s, c_hidden * no_heads; bias=false)
    linear_q_points = PointProjectionMultimer(c_s, no_qk_points, no_heads)
    linear_k_points = PointProjectionMultimer(c_s, no_qk_points, no_heads)
    linear_v_points = PointProjectionMultimer(c_s, no_v_points, no_heads)
    linear_b = LinearFirst(c_z, no_heads)
    head_weights = fill(0.54132485f0, no_heads)
    linear_out = LinearFirst(no_heads * (c_z + c_hidden + no_v_points * 4), c_s)

    return MultimerInvariantPointAttention(
        c_s,
        c_z,
        c_hidden,
        no_heads,
        no_qk_points,
        no_v_points,
        linear_q,
        linear_k,
        linear_v,
        linear_q_points,
        linear_k_points,
        linear_v_points,
        linear_b,
        head_weights,
        linear_out,
        Float32(eps),
        Float32(inf),
    )
end

function (m::MultimerInvariantPointAttention)(s::AbstractArray, z::AbstractArray, r, mask::AbstractArray)
    # s: (C_s, L, B), z: (C_z, L, L, B), mask: (L, B)
    q = m.linear_q(s)
    q = reshape(q, m.c_hidden, m.no_heads, size(q, 2), size(q, 3)) # (C, H, L, B)

    k = m.linear_k(s)
    k = reshape(k, m.c_hidden, m.no_heads, size(k, 2), size(k, 3)) # (C, H, L, B)

    v = m.linear_v(s)
    v = reshape(v, m.c_hidden, m.no_heads, size(v, 2), size(v, 3)) # (C, H, L, B)

    q_bhlc = permutedims(q, (4, 2, 3, 1))
    k_bhlc = permutedims(k, (4, 2, 3, 1))
    k_bhcl = permutedims(k_bhlc, (1, 2, 4, 3))

    B = size(q_bhlc, 1)
    H = size(q_bhlc, 2)
    L = size(q_bhlc, 3)
    C = size(q_bhlc, 4)

    q3 = permutedims(reshape(q_bhlc, B * H, L, C), (2, 3, 1))
    k3 = permutedims(reshape(k_bhcl, B * H, C, L), (2, 3, 1))
    a3 = NNlib.batched_mul(q3, k3)
    a = reshape(a3, L, L, B, H)
    a = permutedims(a, (3, 4, 1, 2)) # (B, H, L, L)
    a = a .* sqrt(1f0 / max(Float32(m.c_hidden), 1f0))

    b = m.linear_b(z) # (H, L, L, B)
    b_perm = permutedims(b, (4, 1, 2, 3))
    a = a .+ b_perm

    q_pts = m.linear_q_points(s, r) # (3, Pq, H, L, B)
    k_pts = m.linear_k_points(s, r) # (3, Pq, H, L, B)
    v_pts = m.linear_v_points(s, r) # (3, Pv, H, L, B)

    q_pts = permutedims(q_pts, (5, 4, 3, 2, 1)) # (B, L, H, Pq, 3)
    k_pts = permutedims(k_pts, (5, 4, 3, 2, 1)) # (B, L, H, Pq, 3)
    v_pts = permutedims(v_pts, (5, 4, 3, 2, 1)) # (B, L, H, Pv, 3)

    q_exp = reshape(q_pts, size(q_pts, 1), size(q_pts, 2), 1, size(q_pts, 3), size(q_pts, 4), size(q_pts, 5))
    k_exp = reshape(k_pts, size(k_pts, 1), 1, size(k_pts, 2), size(k_pts, 3), size(k_pts, 4), size(k_pts, 5))
    pt_att = q_exp .- k_exp
    pt_att = sum(pt_att .^ 2; dims=6)

    head_weights = NNlib.softplus.(m.head_weights)
    point_variance = max(Float32(m.no_qk_points), 1f0) * 9f0 / 2f0
    head_weights = head_weights .* sqrt(1f0 / point_variance)
    hw = reshape(head_weights, 1, 1, 1, m.no_heads, 1, 1)
    pt_att = sum(pt_att .* hw; dims=5) .* (-0.5f0)
    pt_att = dropdims(pt_att; dims=(5, 6))
    pt_att = permutedims(pt_att, (1, 4, 2, 3)) # (B, H, L, L)
    a = a .+ pt_att

    square_mask = reshape(mask, size(mask, 1), 1, size(mask, 2)) .* reshape(mask, 1, size(mask, 1), size(mask, 2))
    square_mask = permutedims(square_mask, (3, 1, 2)) # (B, L, L)
    square_mask = -m.inf .* (1f0 .- square_mask)
    a = a .+ reshape(square_mask, size(square_mask, 1), 1, size(square_mask, 2), size(square_mask, 3))
    a = a .* sqrt(1f0 / 3f0)
    a = NNlib.softmax(a; dims=4)

    v_bhlc = permutedims(v, (4, 2, 3, 1))
    a3 = reshape(permutedims(a, (3, 4, 1, 2)), L, L, :)
    v3 = permutedims(reshape(v_bhlc, B * H, L, C), (2, 3, 1))
    o3 = NNlib.batched_mul(a3, v3)
    o = reshape(o3, L, C, B, H)
    o = permutedims(o, (3, 1, 4, 2)) # (B, L, H, C)
    o = permutedims(o, (1, 2, 4, 3)) # (B, L, C, H)
    o = reshape(o, B, L, H * C)

    v_pts_x = _view_last1(v_pts, 1)
    v_pts_y = _view_last1(v_pts, 2)
    v_pts_z = _view_last1(v_pts, 3)

    vpx = permutedims(v_pts_x, (2, 4, 1, 3))
    vpy = permutedims(v_pts_y, (2, 4, 1, 3))
    vpz = permutedims(v_pts_z, (2, 4, 1, 3))

    vpx3 = reshape(vpx, L, m.no_v_points, :)
    vpy3 = reshape(vpy, L, m.no_v_points, :)
    vpz3 = reshape(vpz, L, m.no_v_points, :)

    o_px = NNlib.batched_mul(a3, vpx3)
    o_py = NNlib.batched_mul(a3, vpy3)
    o_pz = NNlib.batched_mul(a3, vpz3)

    o_px = reshape(o_px, L, m.no_v_points, B, m.no_heads)
    o_py = reshape(o_py, L, m.no_v_points, B, m.no_heads)
    o_pz = reshape(o_pz, L, m.no_v_points, B, m.no_heads)

    o_px = permutedims(o_px, (3, 1, 4, 2))
    o_py = permutedims(o_py, (3, 1, 4, 2))
    o_pz = permutedims(o_pz, (3, 1, 4, 2))

    o_pt = cat(o_px, o_py, o_pz; dims=5) # (B, L, H, P, 3)
    o_pt = invert_apply_rigid(r, permutedims(o_pt, (5, 4, 3, 2, 1))) # (3, P, H, L, B)
    o_pt = permutedims(o_pt, (5, 4, 3, 2, 1)) # (B, L, H, P, 3)

    o_pt_norm = sqrt.(sum(o_pt .^ 2; dims=5) .+ m.eps)
    o_pt_norm = dropdims(o_pt_norm; dims=5)
    o_pt_norm = permutedims(o_pt_norm, (1, 2, 4, 3)) # (B, L, P, H)
    o_pt_norm = reshape(o_pt_norm, size(o_pt_norm, 1), size(o_pt_norm, 2), m.no_heads * m.no_v_points)

    o_pt = permutedims(o_pt, (1, 2, 4, 3, 5))
    o_pt = reshape(o_pt, size(o_pt)[1:end-3]..., m.no_heads * m.no_v_points, 3)
    o_px = _view_last1(o_pt, 1)
    o_py = _view_last1(o_pt, 2)
    o_pz = _view_last1(o_pt, 3)

    a_t = permutedims(a, (1, 2, 4, 3))
    a_exp = reshape(a_t, size(a_t, 1), size(a_t, 2), size(a_t, 3), size(a_t, 4), 1)
    z_swap = permutedims(z, (4, 3, 2, 1))
    z_exp = reshape(z_swap, size(z_swap, 1), 1, size(z_swap, 2), size(z_swap, 3), size(z_swap, 4))
    o_pair = sum(a_exp .* z_exp; dims=3)
    o_pair = dropdims(o_pair; dims=3)
    o_pair = permutedims(o_pair, (1, 3, 2, 4)) # (B, L, H, C_z)
    o_pair = permutedims(o_pair, (1, 2, 4, 3)) # (B, L, C_z, H)
    o_pair = reshape(o_pair, size(o_pair, 1), size(o_pair, 2), m.no_heads * m.c_z)

    o_feat = permutedims(o, (3, 2, 1))
    o_px = permutedims(o_px, (3, 2, 1))
    o_py = permutedims(o_py, (3, 2, 1))
    o_pz = permutedims(o_pz, (3, 2, 1))
    o_pt_norm = permutedims(o_pt_norm, (3, 2, 1))
    o_pair = permutedims(o_pair, (3, 2, 1))

    concat = cat(o_feat, o_px, o_py, o_pz, o_pt_norm, o_pair; dims=1)
    return m.linear_out(concat)
end

@concrete struct BackboneUpdate <: Onion.Layer
    linear
end

@layer BackboneUpdate

function BackboneUpdate(c_s::Int)
    return BackboneUpdate(LinearFirst(c_s, 6))
end

(m::BackboneUpdate)(s) = m.linear(s)

@concrete struct StructureModuleTransitionLayer <: Onion.Layer
    linear_1
    linear_2
    linear_3
end

@layer StructureModuleTransitionLayer

function StructureModuleTransitionLayer(c::Int)
    linear_1 = LinearFirst(c, c)
    linear_2 = LinearFirst(c, c)
    linear_3 = LinearFirst(c, c)
    return StructureModuleTransitionLayer(linear_1, linear_2, linear_3)
end

function (m::StructureModuleTransitionLayer)(s)
    s0 = s
    s = m.linear_1(s)
    s = max.(s, 0f0)
    s = m.linear_2(s)
    s = max.(s, 0f0)
    s = m.linear_3(s)
    return s .+ s0
end

@concrete struct StructureModuleTransition <: Onion.Layer
    layers
    dropout
    layer_norm
end

@layer StructureModuleTransition

function StructureModuleTransition(c::Int, num_layers::Int, dropout_rate::Real)
    layers = [StructureModuleTransitionLayer(c) for _ in 1:num_layers]
    drop = SharedDropout(dropout_rate, 3)
    layer_norm = LayerNormFirst(c)
    return StructureModuleTransition(layers, drop, layer_norm)
end

function (m::StructureModuleTransition)(s)
    for l in m.layers
        s = l(s)
    end
    s = m.dropout(s)
    s = m.layer_norm(s)
    return s
end

@concrete struct AngleResnetBlock <: Onion.Layer
    linear_1
    linear_2
end

@layer AngleResnetBlock

function AngleResnetBlock(c_hidden::Int)
    linear_1 = LinearFirst(c_hidden, c_hidden)
    linear_2 = LinearFirst(c_hidden, c_hidden)
    return AngleResnetBlock(linear_1, linear_2)
end

function (m::AngleResnetBlock)(a)
    s = a
    a = m.linear_1(max.(a, 0f0))
    a = m.linear_2(max.(a, 0f0))
    return a .+ s
end

@concrete struct AngleResnet <: Onion.Layer
    linear_in
    linear_initial
    layers
    linear_out
    eps::Float32
end

@layer AngleResnet

function AngleResnet(c_in::Int, c_hidden::Int, no_blocks::Int, no_angles::Int, epsilon::Float32)
    linear_in = LinearFirst(c_in, c_hidden)
    linear_initial = LinearFirst(c_in, c_hidden)
    layers = [AngleResnetBlock(c_hidden) for _ in 1:no_blocks]
    linear_out = LinearFirst(c_hidden, no_angles * 2)
    return AngleResnet(linear_in, linear_initial, layers, linear_out, Float32(epsilon))
end

function _reshape_first_corder(x::AbstractArray, d1::Int, d2::Int)
    x_perm = permutedims(x, (2:ndims(x)..., 1))
    y = reshape(x_perm, size(x_perm)[1:end-1]..., d2, d1)
    perm = vcat(ndims(y) - 1, ndims(y), collect(1:(ndims(y) - 2)))
    return permutedims(y, perm)
end

function (m::AngleResnet)(s, s_initial)
    s_initial = max.(s_initial, 0f0)
    s_initial = m.linear_initial(s_initial)
    s = max.(s, 0f0)
    s = m.linear_in(s)
    s = s .+ s_initial
    for l in m.layers
        s = l(s)
    end
    s = max.(s, 0f0)
    s = m.linear_out(s)
    s = _reshape_first_corder(s, div(size(s, 1), 2), 2)
    unnormalized = s
    norm = sqrt.(max.(sum(s .^ 2; dims=1), m.eps))
    s = s ./ norm
    return unnormalized, s
end

@concrete struct StructureModule <: Onion.Layer
    cfg::StructureModuleConfig
    layer_norm_s
    layer_norm_z
    linear_in
    ipa
    ipa_dropout
    layer_norm_ipa
    transition
    bb_update
    angle_resnet
    default_frames::Any
    group_idx::Any
    atom_mask::Any
    lit_positions::Any
end

@layer StructureModule

function _init_residue_constants!(m::StructureModule, like)
    default_frames = to_device(restype_rigid_group_default_frame, like, eltype(like))
    group_idx = to_device(restype_atom14_to_rigid_group, like, Int)
    atom_mask = to_device(restype_atom14_mask, like, eltype(like))
    lit_positions = to_device(restype_atom14_rigid_group_positions, like, eltype(like))
    return default_frames, group_idx, atom_mask, lit_positions
end

function StructureModule(; cfg::StructureModuleConfig=StructureModuleConfig())
    layer_norm_s = LayerNormFirst(cfg.c_s)
    layer_norm_z = LayerNormFirst(cfg.c_z)
    linear_in = LinearFirst(cfg.c_s, cfg.c_s)
    ipa = ESMFoldIPA(
        cfg.c_s, cfg.c_z, cfg.c_ipa, cfg.no_heads_ipa,
        cfg.no_qk_points, cfg.no_v_points;
        inf=cfg.inf, eps=cfg.epsilon,
    )
    ipa_dropout = SharedDropout(cfg.dropout_rate, 3)
    layer_norm_ipa = LayerNormFirst(cfg.c_s)
    transition = StructureModuleTransition(cfg.c_s, cfg.no_transition_layers, cfg.dropout_rate)
    bb_update = BackboneUpdate(cfg.c_s)
    angle_resnet = AngleResnet(cfg.c_s, cfg.c_resnet, cfg.no_resnet_blocks, cfg.no_angles, cfg.epsilon)

    return StructureModule(
        cfg, layer_norm_s, layer_norm_z, linear_in,
        ipa, ipa_dropout, layer_norm_ipa, transition,
        bb_update, angle_resnet, nothing, nothing, nothing, nothing,
    )
end

function (m::StructureModule)(evoformer_output_dict, aatype, mask=nothing)
    s = evoformer_output_dict[:single]
    z = evoformer_output_dict[:pair]

    if mask === nothing
        mask = ones_like(s, size(s, 2), size(s, 3))
    end

    s = m.layer_norm_s(s)
    z = m.layer_norm_z(z)

    s_initial = s
    s = m.linear_in(s)

    rigids = rigid_identity((size(s, 2), size(s, 3)), s; fmt=:quat)

    outputs = Zygote.Buffer(Vector{Dict{Symbol,Any}}(undef, m.cfg.no_blocks))
    for i in 1:m.cfg.no_blocks
        s = s .+ m.ipa(s, z, rigids, mask)
        s = m.ipa_dropout(s)
        s = m.layer_norm_ipa(s)
        s = m.transition(s)

        rigids = compose_q_update_vec(rigids, m.bb_update(s))

        backb_to_global = Rigid(RotMatRotation(get_rot_mats(rigids.rots)), rigids.trans)
        backb_to_global = scale_translation(backb_to_global, m.cfg.trans_scale_factor)

        unnormalized_angles, angles = m.angle_resnet(s, s_initial)

        default_frames, group_idx, atom_mask, lit_positions = _init_residue_constants!(m, angles)
        all_frames_to_global = torsion_angles_to_frames(backb_to_global, angles, aatype, default_frames)
        pred_xyz = frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global, aatype, default_frames,
            group_idx, atom_mask, lit_positions,
        )

        scaled_rigids = scale_translation(rigids, m.cfg.trans_scale_factor)

        preds = Dict{Symbol,Any}(
            :frames => to_tensor_7(scaled_rigids),
            :sidechain_frames => to_tensor_4x4(all_frames_to_global),
            :unnormalized_angles => unnormalized_angles,
            :angles => angles,
            :positions => pred_xyz,
            :states => s,
        )
        outputs[i] = preds
    end

    out = stack_dicts(copy(outputs))
    out[:single] = s
    return out
end
