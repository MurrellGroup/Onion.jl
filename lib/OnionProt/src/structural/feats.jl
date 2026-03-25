# ──── Frame/position conversion (OpenFold) ────

function rigid_from_tensor_4x4(t::AbstractArray)
    rot = _view_first2(t, 1:3, 1:3)
    trans = _view_first2(t, 1:3, 4)
    return Rigid(RotMatRotation(rot), trans)
end

function torsion_angles_to_frames(r::Rigid, alpha::AbstractArray, aatype::AbstractArray, default_frames::AbstractArray)
    idx = aatype .+ 1
    df = permutedims(default_frames, (2, 3, 4, 1))
    df_sel = NNlib.gather(df, idx)
    default_4x4 = permutedims(df_sel, (2, 3, 1, 4, 5))

    default_r = rigid_from_tensor_4x4(default_4x4)

    bb_zero = zeros_like(alpha, eltype(alpha), 1, 1, size(alpha, 3), size(alpha, 4))
    bb_one = ones_like(alpha, eltype(alpha), 1, 1, size(alpha, 3), size(alpha, 4))
    bb_rot = cat(bb_zero, bb_one; dims=1)
    alpha = cat(bb_rot, alpha; dims=2)

    a = _view_first1(alpha, 1)
    b = _view_first1(alpha, 2)
    a1 = reshape(a, 1, size(a)...)
    b1 = reshape(b, 1, size(b)...)
    z = zeros_like(alpha, eltype(alpha), 1, size(a)...)
    o = ones_like(alpha, eltype(alpha), 1, size(a)...)
    col1 = reshape(cat(o, z, z, z; dims=1), 4, 1, size(a)...)
    col2 = reshape(cat(z, b1, a1, z; dims=1), 4, 1, size(a)...)
    col3 = reshape(cat(z, -a1, b1, z; dims=1), 4, 1, size(a)...)
    col4 = reshape(cat(z, z, z, o; dims=1), 4, 1, size(a)...)
    all_rots = cat(col1, col2, col3, col4; dims=2)

    all_rots_r = rigid_from_tensor_4x4(all_rots)
    all_frames = compose(default_r, all_rots_r)

    chi2_frame_to_frame = rigid_index(all_frames, 6, Colon(), Colon())
    chi3_frame_to_frame = rigid_index(all_frames, 7, Colon(), Colon())
    chi4_frame_to_frame = rigid_index(all_frames, 8, Colon(), Colon())

    chi1_frame_to_bb = rigid_index(all_frames, 5, Colon(), Colon())
    chi2_frame_to_bb = compose(chi1_frame_to_bb, chi2_frame_to_frame)
    chi3_frame_to_bb = compose(chi2_frame_to_bb, chi3_frame_to_frame)
    chi4_frame_to_bb = compose(chi3_frame_to_bb, chi4_frame_to_frame)

    rot = get_rot_mats(all_frames.rots)
    trans = all_frames.trans
    rot_first = rot[:, :, 1:5, :, :]
    trans_first = trans[:, 1:5, :, :]
    rot_chi2 = reshape(get_rot_mats(chi2_frame_to_bb.rots), 3, 3, 1, size(rot, 4), size(rot, 5))
    rot_chi3 = reshape(get_rot_mats(chi3_frame_to_bb.rots), 3, 3, 1, size(rot, 4), size(rot, 5))
    rot_chi4 = reshape(get_rot_mats(chi4_frame_to_bb.rots), 3, 3, 1, size(rot, 4), size(rot, 5))
    trans_chi2 = reshape(chi2_frame_to_bb.trans, 3, 1, size(trans, 3), size(trans, 4))
    trans_chi3 = reshape(chi3_frame_to_bb.trans, 3, 1, size(trans, 3), size(trans, 4))
    trans_chi4 = reshape(chi4_frame_to_bb.trans, 3, 1, size(trans, 3), size(trans, 4))
    rot_new = cat(rot_first, rot_chi2, rot_chi3, rot_chi4; dims=3)
    trans_new = cat(trans_first, trans_chi2, trans_chi3, trans_chi4; dims=2)
    all_frames_to_bb = Rigid(RotMatRotation(rot_new), trans_new)

    all_frames_to_global = compose(r, all_frames_to_bb)
    return all_frames_to_global
end

function frames_and_literature_positions_to_atom14_pos(
    r::Rigid,
    aatype::AbstractArray,
    default_frames::AbstractArray,
    group_idx::AbstractArray,
    atom_mask::AbstractArray,
    lit_positions::AbstractArray,
)
    group_mask, lit_pos_sel, atom_mask_sel = @ignore_derivatives begin
        idx = aatype .+ 1
        g = permutedims(group_idx, (2, 1))
        group_sel = NNlib.gather(g, idx)

        am = permutedims(atom_mask, (2, 1))
        mask_sel = NNlib.gather(am, idx)
        _atom_mask_sel = reshape(mask_sel, 1, size(mask_sel)...)

        lp = permutedims(lit_positions, (2, 3, 1))
        lp_sel = NNlib.gather(lp, idx)
        _lit_pos_sel = permutedims(lp_sel, (2, 1, 3, 4))

        gm = one_hot_last(group_sel, size(default_frames, 2))
        gm = permutedims(gm, (4, 1, 2, 3))
        gm = convert.(eltype(r.trans), gm)

        (gm, _lit_pos_sel, _atom_mask_sel)
    end

    rot = get_rot_mats(r.rots)
    trans = r.trans

    rot_exp = reshape(rot, 3, 3, 8, 1, size(rot, 4), size(rot, 5))
    mask_exp = reshape(group_mask, 1, 1, 8, 14, size(group_mask, 3), size(group_mask, 4))
    rot_atom = sum(rot_exp .* mask_exp; dims=3)
    rot_atom = dropdims(rot_atom; dims=3)

    trans_exp = reshape(trans, 3, 8, 1, size(trans, 3), size(trans, 4))
    mask_t = reshape(group_mask, 1, 8, 14, size(group_mask, 3), size(group_mask, 4))
    trans_atom = sum(trans_exp .* mask_t; dims=2)
    trans_atom = dropdims(trans_atom; dims=2)

    pred = rot_vec_mul_first(rot_atom, lit_pos_sel) .+ trans_atom
    pred = pred .* atom_mask_sel
    return pred
end

function atom14_to_atom37(atom14::AbstractArray, batch::AbstractDict)
    idx = batch[:residx_atom37_to_atom14]

    if ndims(idx) == 2
        L = size(atom14, 3)
        idx = reshape(idx, L, 1, size(idx, 2))
        idx = repeat(idx, 1, size(atom14, 4), 1)
    elseif ndims(idx) == 3 && size(idx, 1) == 37
        idx = permutedims(idx, (2, 3, 1))
    end

    idx1 = idx .+ 1

    src = permutedims(atom14, (1, 4, 3, 2))

    B, L_dim, A = size(idx1, 2), size(idx1, 1), size(idx1, 3)
    idx_cart = Array{CartesianIndex{3}}(undef, B, L_dim, A)
    for b in 1:B, l in 1:L_dim, a in 1:A
        idx_cart[b, l, a] = CartesianIndex(b, l, idx1[l, b, a])
    end
    idx_cart = to_device(idx_cart, atom14, CartesianIndex{3})

    gathered = NNlib.gather(src, idx_cart)
    out = permutedims(gathered, (2, 3, 4, 1))

    atom37_exists = batch[:atom37_atom_exists]
    if ndims(atom37_exists) == 2
        atom37_exists = reshape(atom37_exists, size(atom37_exists, 1), 1, size(atom37_exists, 2))
        atom37_exists = repeat(atom37_exists, 1, size(atom14, 4), 1)
        atom37_exists = permutedims(atom37_exists, (2, 1, 3))
    elseif ndims(atom37_exists) == 3 && size(atom37_exists, 1) == 37
        atom37_exists = permutedims(atom37_exists, (3, 2, 1))
    else
        atom37_exists = permutedims(atom37_exists, (2, 1, 3))
    end
    out = out .* reshape(atom37_exists, size(atom37_exists)..., 1)
    return out
end
