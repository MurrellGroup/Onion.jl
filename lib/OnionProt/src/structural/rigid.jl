# ──── Rigid body types + operations ────
# SE(3) representation matching OpenFold conventions.
# Replaces old BatchedTransformations.jl dependency with direct operations.

@inline _view_last2(x, I, J) = view(x, ntuple(_ -> Colon(), ndims(x) - 2)..., I, J)
@inline _view_last1(x, I) = view(x, ntuple(_ -> Colon(), ndims(x) - 1)..., I)
@inline _view_first2(x, I, J) = view(x, I, J, ntuple(_ -> Colon(), ndims(x) - 2)...)
@inline _view_first1(x, I) = view(x, I, ntuple(_ -> Colon(), ndims(x) - 1)...)

abstract type AbstractRotation end

struct RotMatRotation{T<:AbstractArray} <: AbstractRotation
    rot_mats::T
end

struct QuatRotation{T<:AbstractArray} <: AbstractRotation
    quats::T
end

rot_from_mat(rot_mats::AbstractArray) = RotMatRotation(rot_mats)

function rot_from_quat(quats::AbstractArray; normalize::Bool=true)
    q = normalize ? quats ./ sqrt.(sum(quats .^ 2; dims=1)) : quats
    return QuatRotation(q)
end

function rot_matmul_first(a::AbstractArray, b::AbstractArray)
    a11 = _view_first2(a, 1, 1); a12 = _view_first2(a, 1, 2); a13 = _view_first2(a, 1, 3)
    a21 = _view_first2(a, 2, 1); a22 = _view_first2(a, 2, 2); a23 = _view_first2(a, 2, 3)
    a31 = _view_first2(a, 3, 1); a32 = _view_first2(a, 3, 2); a33 = _view_first2(a, 3, 3)

    b11 = _view_first2(b, 1, 1); b12 = _view_first2(b, 1, 2); b13 = _view_first2(b, 1, 3)
    b21 = _view_first2(b, 2, 1); b22 = _view_first2(b, 2, 2); b23 = _view_first2(b, 2, 3)
    b31 = _view_first2(b, 3, 1); b32 = _view_first2(b, 3, 2); b33 = _view_first2(b, 3, 3)

    c11 = a11 .* b11 .+ a12 .* b21 .+ a13 .* b31
    c12 = a11 .* b12 .+ a12 .* b22 .+ a13 .* b32
    c13 = a11 .* b13 .+ a12 .* b23 .+ a13 .* b33

    c21 = a21 .* b11 .+ a22 .* b21 .+ a23 .* b31
    c22 = a21 .* b12 .+ a22 .* b22 .+ a23 .* b32
    c23 = a21 .* b13 .+ a22 .* b23 .+ a23 .* b33

    c31 = a31 .* b11 .+ a32 .* b21 .+ a33 .* b31
    c32 = a31 .* b12 .+ a32 .* b22 .+ a33 .* b32
    c33 = a31 .* b13 .+ a32 .* b23 .+ a33 .* b33

    r1 = cat(reshape(c11, 1, 1, size(c11)...), reshape(c12, 1, 1, size(c12)...), reshape(c13, 1, 1, size(c13)...); dims=2)
    r2 = cat(reshape(c21, 1, 1, size(c21)...), reshape(c22, 1, 1, size(c22)...), reshape(c23, 1, 1, size(c23)...); dims=2)
    r3 = cat(reshape(c31, 1, 1, size(c31)...), reshape(c32, 1, 1, size(c32)...), reshape(c33, 1, 1, size(c33)...); dims=2)
    return cat(r1, r2, r3; dims=1)
end

function rot_vec_mul_first(r::AbstractArray, t::AbstractArray)
    x = _view_first1(t, 1)
    y = _view_first1(t, 2)
    z = _view_first1(t, 3)
    o1 = _view_first2(r, 1, 1) .* x .+ _view_first2(r, 1, 2) .* y .+ _view_first2(r, 1, 3) .* z
    o2 = _view_first2(r, 2, 1) .* x .+ _view_first2(r, 2, 2) .* y .+ _view_first2(r, 2, 3) .* z
    o3 = _view_first2(r, 3, 1) .* x .+ _view_first2(r, 3, 2) .* y .+ _view_first2(r, 3, 3) .* z
    return cat(reshape(o1, 1, size(o1)...), reshape(o2, 1, size(o2)...), reshape(o3, 1, size(o3)...); dims=1)
end

function quat_multiply_first(q1::AbstractArray, q2::AbstractArray)
    a1 = _view_first1(q1, 1); b1 = _view_first1(q1, 2); c1 = _view_first1(q1, 3); d1 = _view_first1(q1, 4)
    a2 = _view_first1(q2, 1); b2 = _view_first1(q2, 2); c2 = _view_first1(q2, 3); d2 = _view_first1(q2, 4)
    w = a1 .* a2 .- b1 .* b2 .- c1 .* c2 .- d1 .* d2
    x = a1 .* b2 .+ b1 .* a2 .+ c1 .* d2 .- d1 .* c2
    y = a1 .* c2 .- b1 .* d2 .+ c1 .* a2 .+ d1 .* b2
    z = a1 .* d2 .+ b1 .* c2 .- c1 .* b2 .+ d1 .* a2
    return cat(reshape(w, 1, size(w)...), reshape(x, 1, size(x)...), reshape(y, 1, size(y)...), reshape(z, 1, size(z)...); dims=1)
end

function quat_multiply_by_vec_first(q::AbstractArray, v::AbstractArray)
    a = _view_first1(q, 1); b = _view_first1(q, 2); c = _view_first1(q, 3); d = _view_first1(q, 4)
    vx = _view_first1(v, 1); vy = _view_first1(v, 2); vz = _view_first1(v, 3)
    w = -b .* vx .- c .* vy .- d .* vz
    x = a .* vx .+ c .* vz .- d .* vy
    y = a .* vy .- b .* vz .+ d .* vx
    z = a .* vz .+ b .* vy .- c .* vx
    return cat(reshape(w, 1, size(w)...), reshape(x, 1, size(x)...), reshape(y, 1, size(y)...), reshape(z, 1, size(z)...); dims=1)
end

function quat_to_rot_first(quat::AbstractArray)
    q = quat ./ sqrt.(sum(quat .^ 2; dims=1))
    w = _view_first1(q, 1); x = _view_first1(q, 2); y = _view_first1(q, 3); z = _view_first1(q, 4)

    ww = w .* w; xx = x .* x; yy = y .* y; zz = z .* z
    wx = w .* x; wy = w .* y; wz = w .* z
    xy = x .* y; xz = x .* z; yz = y .* z

    r11 = 1 .- 2 .* (yy .+ zz)
    r12 = 2 .* (xy .- wz)
    r13 = 2 .* (xz .+ wy)
    r21 = 2 .* (xy .+ wz)
    r22 = 1 .- 2 .* (xx .+ zz)
    r23 = 2 .* (yz .- wx)
    r31 = 2 .* (xz .- wy)
    r32 = 2 .* (yz .+ wx)
    r33 = 1 .- 2 .* (xx .+ yy)

    r1 = cat(reshape(r11, 1, 1, size(r11)...), reshape(r12, 1, 1, size(r12)...), reshape(r13, 1, 1, size(r13)...); dims=2)
    r2 = cat(reshape(r21, 1, 1, size(r21)...), reshape(r22, 1, 1, size(r22)...), reshape(r23, 1, 1, size(r23)...); dims=2)
    r3 = cat(reshape(r31, 1, 1, size(r31)...), reshape(r32, 1, 1, size(r32)...), reshape(r33, 1, 1, size(r33)...); dims=2)
    return cat(r1, r2, r3; dims=1)
end

function rotation_identity(shape::Tuple, like::AbstractArray; fmt::Symbol=:quat)
    if fmt == :rot_mat
        eye = ones_like(like, eltype(like), 3, 3)
        eye = @ignore_derivatives begin
            m = zeros_like(like, eltype(like), 3, 3)
            m[1,1] = one(eltype(like))
            m[2,2] = one(eltype(like))
            m[3,3] = one(eltype(like))
            m
        end
        eye = reshape(eye, 3, 3, ntuple(_ -> 1, length(shape))...)
        rot = repeat(eye, 1, 1, shape...)
        return RotMatRotation(rot)
    elseif fmt == :quat
        z = zeros_like(like, eltype(like), 1, shape...)
        o = ones_like(like, eltype(like), 1, shape...)
        q = cat(o, z, z, z; dims=1)
        return QuatRotation(q)
    else
        error("Unknown rotation format: $fmt")
    end
end

get_rot_mats(r::RotMatRotation) = r.rot_mats
get_rot_mats(r::QuatRotation) = quat_to_rot_first(r.quats)

get_quats(r::QuatRotation) = r.quats
get_quats(::RotMatRotation) = error("rot_to_quat_first not implemented")

function compose_q_update_vec(r::QuatRotation, q_update_vec::AbstractArray; normalize_quats::Bool=true)
    q = r.quats
    new_quats = q .+ quat_multiply_by_vec_first(q, q_update_vec)
    if normalize_quats
        new_quats = new_quats ./ sqrt.(sum(new_quats .^ 2; dims=1))
    end
    return QuatRotation(new_quats)
end

struct Rigid{R<:AbstractRotation,T<:AbstractArray}
    rots::R
    trans::T
end

function rigid_identity(shape::Tuple, like::AbstractArray; fmt::Symbol=:quat)
    rots = rotation_identity(shape, like; fmt=fmt)
    trans = zeros_like(like, eltype(like), 3, shape...)
    return Rigid(rots, trans)
end

function apply_rotation(r::AbstractRotation, pts::AbstractArray)
    rot = get_rot_mats(r)
    return rot_vec_mul_first(rot, pts)
end

function apply_rigid(r::Rigid, pts::AbstractArray)
    rot = get_rot_mats(r.rots)
    trans = r.trans
    batch_dims = ndims(trans) - 1
    extra = ndims(pts) - (batch_dims + 1)
    extra < 0 && error("apply_rigid: point rank $(ndims(pts)) incompatible with trans rank $(ndims(trans))")

    if extra > 0
        # Insert singleton dims so rot/trans broadcast over extra point dims.
        # rot: (3, 3, batch...) → (3, 3, 1, ..., batch...)
        # trans: (3, batch...) → (3, 1, ..., batch...)
        rot = reshape(rot, 3, 3, ntuple(_ -> 1, extra)..., size(rot)[3:end]...)
        trans = reshape(trans, 3, ntuple(_ -> 1, extra)..., size(trans)[2:end]...)
    end

    return rot_vec_mul_first(rot, pts) .+ trans
end

function invert_apply_rigid(r::Rigid, pts::AbstractArray)
    rot = get_rot_mats(r.rots)
    trans = r.trans
    batch_dims = ndims(trans) - 1
    extra = ndims(pts) - (batch_dims + 1)
    extra < 0 && error("invert_apply_rigid: point rank $(ndims(pts)) incompatible with trans rank $(ndims(trans))")

    if extra > 0
        rot = reshape(rot, 3, 3, ntuple(_ -> 1, extra)..., size(rot)[3:end]...)
        trans = reshape(trans, 3, ntuple(_ -> 1, extra)..., size(trans)[2:end]...)
    end

    # Rᵀ·(pts - t)
    rotT = permutedims(rot, (2, 1, ntuple(i -> i + 2, ndims(rot) - 2)...))
    return rot_vec_mul_first(rotT, pts .- trans)
end

function compose_q_update_vec(r::Rigid, q_update_vec::AbstractArray)
    q_vec = _view_first1(q_update_vec, 1:3)
    t_vec = _view_first1(q_update_vec, 4:6)
    new_rots = compose_q_update_vec(r.rots, q_vec)
    trans_update = apply_rotation(r.rots, t_vec)
    new_trans = r.trans .+ trans_update
    return Rigid(new_rots, new_trans)
end

scale_translation(r::Rigid, factor::Real) = Rigid(r.rots, r.trans .* factor)

function to_tensor_7(r::Rigid)
    q = get_quats(r.rots)
    return cat(q, r.trans; dims=1)
end

function to_tensor_4x4(r::Rigid)
    rot = get_rot_mats(r.rots)
    trans = r.trans
    trans_row = reshape(trans, 3, 1, size(trans)[2:end]...)
    top = cat(rot, trans_row; dims=2)
    bottom_zeros = zeros_like(trans, eltype(trans), 1, 3, size(trans)[2:end]...)
    bottom_ones = ones_like(trans, eltype(trans), 1, 1, size(trans)[2:end]...)
    bottom = cat(bottom_zeros, bottom_ones; dims=2)
    return cat(top, bottom; dims=1)
end

function _align_batch_first(a::AbstractArray, target_batch_dims::Int)
    batch_dims = ndims(a) - 2
    if batch_dims < target_batch_dims
        extra = target_batch_dims - batch_dims
        return reshape(a, size(a, 1), size(a, 2), ntuple(_ -> 1, extra)..., size(a)[3:end]...)
    end
    return a
end

function _align_trans_first(a::AbstractArray, target_batch_dims::Int)
    batch_dims = ndims(a) - 1
    if batch_dims < target_batch_dims
        extra = target_batch_dims - batch_dims
        return reshape(a, size(a, 1), ntuple(_ -> 1, extra)..., size(a)[2:end]...)
    end
    return a
end

function compose(r1::Rigid, r2::Rigid)
    rot1 = get_rot_mats(r1.rots)
    rot2 = get_rot_mats(r2.rots)
    batch_dims = max(ndims(rot1) - 2, ndims(rot2) - 2)
    rot1 = _align_batch_first(rot1, batch_dims)
    rot2 = _align_batch_first(rot2, batch_dims)
    trans1 = _align_trans_first(r1.trans, batch_dims)
    trans2 = _align_trans_first(r2.trans, batch_dims)
    new_rot = rot_matmul_first(rot1, rot2)
    new_trans = rot_vec_mul_first(rot1, trans2) .+ trans1
    return Rigid(RotMatRotation(new_rot), new_trans)
end

function rigid_index(r::Rigid, inds...)
    rot = get_rot_mats(r.rots)
    rot = view(rot, :, :, inds...)
    trans = view(r.trans, :, inds...)
    return Rigid(RotMatRotation(rot), trans)
end
