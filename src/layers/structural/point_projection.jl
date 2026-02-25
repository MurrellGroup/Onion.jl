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
    H = m.no_heads
    P = m.num_points
    raw = reshape(raw, P, H, 3, size(raw, 2), size(raw, 3))
    points_local = permutedims(raw, (3, 1, 2, 4, 5))
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
    P = m.num_points
    H = m.no_heads
    raw = reshape(raw, 3 * P, H, size(raw, 2), size(raw, 3))  # (3P, H, L, B)

    x = view(raw, 1:P, :, :, :)
    y = view(raw, (P + 1):(2P), :, :, :)
    z = view(raw, (2P + 1):(3P), :, :, :)

    points_local = cat(
        reshape(x, 1, P, H, size(raw, 3), size(raw, 4)),
        reshape(y, 1, P, H, size(raw, 3), size(raw, 4)),
        reshape(z, 1, P, H, size(raw, 3), size(raw, 4));
        dims=1,
    )  # (3, P, H, L, B)

    points_global = apply_rigid(rigids, points_local)
    return points_global
end
