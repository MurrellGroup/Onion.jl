# ──── Point Projections for IPA ────

"""
    PointProjection(c_hidden, num_points, no_heads)

Projects activations to 3D point clouds, then transforms to global frame via `apply_rigid`.
"""
@concrete struct PointProjection <: Layer
    linear
    num_points::Int
    no_heads::Int
end

function PointProjection(c_hidden::Int, num_points::Int, no_heads::Int)
    linear = Linear(c_hidden => no_heads * 3 * num_points)
    return PointProjection(linear, num_points, no_heads)
end

function (m::PointProjection)(activations, rigids)
    raw = m.linear(activations)
    points_local = rearrange(raw, einops"(P H xyz) L B -> xyz P H L B"; P=m.num_points, H=m.no_heads, xyz=3)
    points_global = apply_rigid(rigids, points_local)
    return points_global
end

"""
    PointProjectionMultimer(c_hidden, num_points, no_heads)

Like `PointProjection` but with `(3P, H)` weight layout (split x/y/z in first dim).
Used by `MultimerInvariantPointAttention`.
"""
@concrete struct PointProjectionMultimer <: Layer
    linear
    num_points::Int
    no_heads::Int
end

function PointProjectionMultimer(c_hidden::Int, num_points::Int, no_heads::Int)
    linear = Linear(c_hidden => no_heads * 3 * num_points)
    return PointProjectionMultimer(linear, num_points, no_heads)
end

function (m::PointProjectionMultimer)(activations::AbstractArray, rigids)
    raw = m.linear(activations)  # (3*P*H, L, B)
    points_local = rearrange(raw, einops"(xyz P H) L B -> xyz P H L B"; xyz=3, P=m.num_points)
    points_global = apply_rigid(rigids, points_local)
    return points_global
end
