# ──── ESMFoldIPA (Invariant Point Attention) ────

using NNlib: batched_mul, softplus

"""
    ESMFoldIPA(c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points; inf=1e5, eps=1e-8)

Invariant Point Attention from ESMFold / OpenFold.
Combines scalar QK attention, 3D point distance attention, and pair bias.
"""
@concrete struct ESMFoldIPA <: Layer
    c_s::Int; c_z::Int; c_hidden::Int
    no_heads::Int; no_qk_points::Int; no_v_points::Int
    linear_q; linear_q_points
    linear_kv; linear_kv_points
    linear_b; head_weights; linear_out
    eps::Float32; inf::Float32
end

function ESMFoldIPA(
    c_s::Int, c_z::Int, c_hidden::Int, no_heads::Int,
    no_qk_points::Int, no_v_points::Int;
    inf::Real=1e5, eps::Real=1e-8,
)
    linear_q = Linear(c_s => c_hidden * no_heads)
    linear_q_points = PointProjection(c_s, no_qk_points, no_heads)
    linear_kv = Linear(c_s => 2 * c_hidden * no_heads)
    linear_kv_points = PointProjection(c_s, no_qk_points + no_v_points, no_heads)
    linear_b = Linear(c_z => no_heads)
    head_weights = fill(0.54132485f0, no_heads)
    linear_out = Linear(no_heads * (c_z + c_hidden + no_v_points * 4) => c_s)
    return ESMFoldIPA(
        c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points,
        linear_q, linear_q_points, linear_kv, linear_kv_points,
        linear_b, head_weights, linear_out, Float32(eps), Float32(inf),
    )
end

function (m::ESMFoldIPA)(s, z, r, mask)
    B = size(s, 3)

    # Scalar Q/K/V projections
    q = rearrange(m.linear_q(s), einops"(C H) L B -> C H L B"; C=m.c_hidden)
    kv = rearrange(m.linear_kv(s), einops"(C H) L B -> C H L B"; C=2*m.c_hidden)
    k = @view kv[1:m.c_hidden, :, :, :]
    v = @view kv[m.c_hidden+1:end, :, :, :]

    # Scalar attention: Q·Kᵀ via batched_mul
    q3 = rearrange(q, einops"C H L B -> L C (B H)")
    k3 = rearrange(k, einops"C H L B -> C L (B H)")
    a = rearrange(batched_mul(q3, k3), einops"Q K (B H) -> B H Q K"; B)
    a = a .* sqrt(1f0 / (3f0 * m.c_hidden))

    # Pair bias
    a = a .+ sqrt(1f0 / 3f0) .* rearrange(m.linear_b(z), einops"H L1 L2 B -> B H L1 L2")

    # Point projections → (B, L, H, P, xyz)
    q_pts = m.linear_q_points(s, r)
    kv_pts = m.linear_kv_points(s, r)
    k_pts = @view kv_pts[:, 1:m.no_qk_points, :, :, :]
    v_pts = @view kv_pts[:, (m.no_qk_points+1):(m.no_qk_points+m.no_v_points), :, :, :]

    q_pts, k_pts, v_pts = rearrange.((q_pts, k_pts, v_pts), einops"xyz P H L B -> B L H P xyz")

    # Point distance attention
    q_exp = rearrange(q_pts, einops"B Q H P xyz -> B Q 1 H P xyz")
    k_exp = rearrange(k_pts, einops"B K H P xyz -> B 1 K H P xyz")
    pt_att = sum((q_exp .- k_exp) .^ 2; dims=6)

    head_weights = softplus.(m.head_weights)
    head_weights = head_weights .* sqrt(1f0 / (3f0 * (m.no_qk_points * 9f0 / 2f0)))
    hw = reshape(head_weights, 1, 1, 1, m.no_heads, 1, 1)
    pt_att = dropdims(sum(pt_att .* hw; dims=5) .* (-0.5f0); dims=(5, 6))
    a = a .+ rearrange(pt_att, einops"B Q K H -> B H Q K")

    # Square mask
    square_mask = rearrange(mask, einops"Q B -> Q 1 B") .* rearrange(mask, einops"K B -> 1 K B")
    square_mask = m.inf .* (rearrange(square_mask, einops"Q K B -> B Q K") .- 1)
    a = a .+ rearrange(square_mask, einops"B Q K -> B 1 Q K")

    a = NNlib.softmax(a; dims=4)

    # Scalar value aggregation
    a3 = rearrange(a, einops"B H Q K -> Q K (B H)")
    v3 = rearrange(v, einops"C H L B -> L C (B H)")
    o = rearrange(batched_mul(a3, v3), einops"L C (B H) -> B L (C H)"; B)

    # Point value aggregation (combined xyz)
    vp = rearrange(v_pts, einops"B L H P xyz -> L (P xyz) (B H)")
    o_pt = rearrange(batched_mul(a3, vp),
        einops"L (P xyz) (B H) -> B L H P xyz"; P=m.no_v_points, xyz=3, B)

    # Apply inverse rigid transform
    o_pt = invert_apply_rigid(r, rearrange(o_pt, einops"B L H P xyz -> xyz P H L B"))
    o_pt = rearrange(o_pt, einops"xyz P H L B -> B L H P xyz")

    # Point norms
    o_pt_norm = sqrt.(sum(o_pt .^ 2; dims=5) .+ m.eps)
    o_pt_norm = rearrange(dropdims(o_pt_norm; dims=5), einops"B L H P -> B L (P H)")

    # Point xyz for output
    o_pt_flat = rearrange(o_pt, einops"B L H P xyz -> B L (P H) xyz")
    o_px = _view_last1(o_pt_flat, 1)
    o_py = _view_last1(o_pt_flat, 2)
    o_pz = _view_last1(o_pt_flat, 3)

    # Pair aggregation via einsum
    o_pair = einsum(a, z, einops"B H Q K, Cz Q K B -> B Q H Cz")
    o_pair = rearrange(o_pair, einops"B Q H Cz -> B Q (Cz H)")

    # Final output: all to (D, L, B) and concatenate
    o, o_px, o_py, o_pz, o_pt_norm, o_pair =
        rearrange.((o, o_px, o_py, o_pz, o_pt_norm, o_pair), einops"B L D -> D L B")

    return m.linear_out(cat(o, o_px, o_py, o_pz, o_pt_norm, o_pair; dims=1))
end

const InvariantPointAttention = ESMFoldIPA

# ──── MultimerInvariantPointAttention ────

"""
    MultimerInvariantPointAttention(c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points; ...)

AF2 multimer IPA. Separate q/k/v projections and PointProjectionMultimer.
"""
@concrete struct MultimerInvariantPointAttention <: Layer
    c_s::Int; c_z::Int; c_hidden::Int
    no_heads::Int; no_qk_points::Int; no_v_points::Int
    linear_q; linear_k; linear_v
    linear_q_points; linear_k_points; linear_v_points
    linear_b; head_weights; linear_out
    eps::Float32; inf::Float32
end

function MultimerInvariantPointAttention(
    c_s::Int, c_z::Int, c_hidden::Int, no_heads::Int,
    no_qk_points::Int, no_v_points::Int;
    inf::Real=1e5, eps::Real=1e-8,
)
    linear_q = Linear(c_s => c_hidden * no_heads, bias=false)
    linear_k = Linear(c_s => c_hidden * no_heads, bias=false)
    linear_v = Linear(c_s => c_hidden * no_heads, bias=false)
    linear_q_points = PointProjectionMultimer(c_s, no_qk_points, no_heads)
    linear_k_points = PointProjectionMultimer(c_s, no_qk_points, no_heads)
    linear_v_points = PointProjectionMultimer(c_s, no_v_points, no_heads)
    linear_b = Linear(c_z => no_heads)
    head_weights = fill(0.54132485f0, no_heads)
    linear_out = Linear(no_heads * (c_z + c_hidden + no_v_points * 4) => c_s)

    return MultimerInvariantPointAttention(
        c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points,
        linear_q, linear_k, linear_v,
        linear_q_points, linear_k_points, linear_v_points,
        linear_b, head_weights, linear_out,
        Float32(eps), Float32(inf),
    )
end

function (m::MultimerInvariantPointAttention)(s::AbstractArray, z::AbstractArray, r, mask::AbstractArray)
    B = size(s, 3)

    # Scalar Q/K/V projections
    q = rearrange(m.linear_q(s), einops"(C H) L B -> C H L B"; C=m.c_hidden)
    k = rearrange(m.linear_k(s), einops"(C H) L B -> C H L B"; C=m.c_hidden)
    v = rearrange(m.linear_v(s), einops"(C H) L B -> C H L B"; C=m.c_hidden)

    # Scalar attention: Q·Kᵀ via batched_mul
    q3 = rearrange(q, einops"C H L B -> L C (B H)")
    k3 = rearrange(k, einops"C H L B -> C L (B H)")
    a = rearrange(batched_mul(q3, k3), einops"Q K (B H) -> B H Q K"; B)
    a = a .* sqrt(1f0 / max(Float32(m.c_hidden), 1f0))

    # Pair bias
    a = a .+ rearrange(m.linear_b(z), einops"H L1 L2 B -> B H L1 L2")

    # Point projections → (B, L, H, P, xyz)
    q_pts, k_pts, v_pts = rearrange.((m.linear_q_points(s, r), m.linear_k_points(s, r), m.linear_v_points(s, r)),
        einops"xyz P H L B -> B L H P xyz")

    # Point distance attention
    q_exp = rearrange(q_pts, einops"B Q H P xyz -> B Q 1 H P xyz")
    k_exp = rearrange(k_pts, einops"B K H P xyz -> B 1 K H P xyz")
    pt_att = sum((q_exp .- k_exp) .^ 2; dims=6)

    head_weights = softplus.(m.head_weights)
    point_variance = max(Float32(m.no_qk_points), 1f0) * 9f0 / 2f0
    head_weights = head_weights .* sqrt(1f0 / point_variance)
    hw = reshape(head_weights, 1, 1, 1, m.no_heads, 1, 1)
    pt_att = dropdims(sum(pt_att .* hw; dims=5) .* (-0.5f0); dims=(5, 6))
    a = a .+ rearrange(pt_att, einops"B Q K H -> B H Q K")

    # Square mask
    square_mask = rearrange(mask, einops"Q B -> Q 1 B") .* rearrange(mask, einops"K B -> 1 K B")
    square_mask = -m.inf .* (1f0 .- rearrange(square_mask, einops"Q K B -> B Q K"))
    a = a .+ rearrange(square_mask, einops"B Q K -> B 1 Q K")
    a = a .* sqrt(1f0 / 3f0)
    a = NNlib.softmax(a; dims=4)

    # Scalar value aggregation
    a3 = rearrange(a, einops"B H Q K -> Q K (B H)")
    v3 = rearrange(v, einops"C H L B -> L C (B H)")
    o = rearrange(batched_mul(a3, v3), einops"L C (B H) -> B L (C H)"; B)

    # Point value aggregation (combined xyz)
    vp = rearrange(v_pts, einops"B L H P xyz -> L (P xyz) (B H)")
    o_pt = rearrange(batched_mul(a3, vp),
        einops"L (P xyz) (B H) -> B L H P xyz"; P=m.no_v_points, xyz=3, B)

    # Apply inverse rigid transform
    o_pt = invert_apply_rigid(r, rearrange(o_pt, einops"B L H P xyz -> xyz P H L B"))
    o_pt = rearrange(o_pt, einops"xyz P H L B -> B L H P xyz")

    # Point norms
    o_pt_norm = sqrt.(sum(o_pt .^ 2; dims=5) .+ m.eps)
    o_pt_norm = rearrange(dropdims(o_pt_norm; dims=5), einops"B L H P -> B L (P H)")

    # Point xyz for output
    o_pt_flat = rearrange(o_pt, einops"B L H P xyz -> B L (P H) xyz")
    o_px = _view_last1(o_pt_flat, 1)
    o_py = _view_last1(o_pt_flat, 2)
    o_pz = _view_last1(o_pt_flat, 3)

    # Pair aggregation via einsum
    o_pair = einsum(a, z, einops"B H Q K, Cz Q K B -> B Q H Cz")
    o_pair = rearrange(o_pair, einops"B Q H Cz -> B Q (Cz H)")

    # Final output: all to (D, L, B) and concatenate
    o, o_px, o_py, o_pz, o_pt_norm, o_pair =
        rearrange.((o, o_px, o_py, o_pz, o_pt_norm, o_pair), einops"B L D -> D L B")

    return m.linear_out(cat(o, o_px, o_py, o_pz, o_pt_norm, o_pair; dims=1))
end
