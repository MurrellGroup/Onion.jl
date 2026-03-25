# ──── StructureModule (OpenFold/ESMFold) ────

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

"""
    BackboneUpdate(c_s)

Linear projection to 6D vector (3 quaternion update + 3 translation update).
"""
@concrete struct BackboneUpdate <: Layer
    linear
end

BackboneUpdate(c_s::Int) = BackboneUpdate(Linear(c_s => 6))
(m::BackboneUpdate)(s) = m.linear(s)

@concrete struct StructureModuleTransitionLayer <: Layer
    linear_1; linear_2; linear_3
end

function StructureModuleTransitionLayer(c::Int)
    return StructureModuleTransitionLayer(
        Linear(c => c), Linear(c => c), Linear(c => c),
    )
end

function (m::StructureModuleTransitionLayer)(s)
    s0 = s
    s = max.(m.linear_1(s), 0f0)
    s = max.(m.linear_2(s), 0f0)
    s = m.linear_3(s)
    return s .+ s0
end

@concrete struct StructureModuleTransition <: Layer
    layers; dropout; layer_norm
end

function StructureModuleTransition(c::Int, num_layers::Int, dropout_rate::Real)
    layers = [StructureModuleTransitionLayer(c) for _ in 1:num_layers]
    drop = SharedDropout(dropout_rate, [3])
    ln = LayerNorm(c)
    return StructureModuleTransition(layers, drop, ln)
end

function (m::StructureModuleTransition)(s)
    for l in m.layers
        s = l(s)
    end
    s = m.dropout(s)
    s = m.layer_norm(s)
    return s
end

"""
    AngleResnetBlock(c_hidden)

2-layer residual block: `relu → linear → relu → linear + residual`.
"""
@concrete struct AngleResnetBlock <: Layer
    linear_1; linear_2
end

AngleResnetBlock(c_hidden::Int) = AngleResnetBlock(Linear(c_hidden => c_hidden), Linear(c_hidden => c_hidden))

function (m::AngleResnetBlock)(a)
    s = a
    a = m.linear_2(max.(m.linear_1(max.(a, 0f0)), 0f0))
    return a .+ s
end

"""
    AngleResnet(c_in, c_hidden, no_blocks, no_angles, epsilon)

Predicts (unnormalized, normalized) torsion angles from two sequence representations.
"""
@concrete struct AngleResnet <: Layer
    linear_in; linear_initial; layers; linear_out
    eps::Float32
end

function AngleResnet(c_in::Int, c_hidden::Int, no_blocks::Int, no_angles::Int, epsilon::Float32)
    linear_in = Linear(c_in => c_hidden)
    linear_initial = Linear(c_in => c_hidden)
    layers = [AngleResnetBlock(c_hidden) for _ in 1:no_blocks]
    linear_out = Linear(c_hidden => no_angles * 2)
    return AngleResnet(linear_in, linear_initial, layers, linear_out, epsilon)
end

function _reshape_first_corder(x::AbstractArray, d1::Int, d2::Int)
    x_perm = permutedims(x, (2:ndims(x)..., 1))
    y = reshape(x_perm, size(x_perm)[1:end-1]..., d2, d1)
    perm = vcat(ndims(y) - 1, ndims(y), collect(1:(ndims(y) - 2)))
    return permutedims(y, perm)
end

function (m::AngleResnet)(s, s_initial)
    s_initial = m.linear_initial(max.(s_initial, 0f0))
    s = m.linear_in(max.(s, 0f0))
    s = s .+ s_initial
    for l in m.layers
        s = l(s)
    end
    s = m.linear_out(max.(s, 0f0))
    s = _reshape_first_corder(s, div(size(s, 1), 2), 2)
    unnormalized = s
    norm = sqrt.(max.(sum(s .^ 2; dims=1), m.eps))
    s = s ./ norm
    return unnormalized, s
end

"""
    StructureModule(; cfg=StructureModuleConfig())

OpenFold/ESMFold structure module. Iteratively refines backbone frames and
predicts sidechain torsion angles + atom14 positions.
"""
@concrete struct StructureModule <: Layer
    cfg::StructureModuleConfig
    layer_norm_s; layer_norm_z; linear_in
    ipa; ipa_dropout; layer_norm_ipa
    transition; bb_update; angle_resnet
end

function StructureModule(; cfg::StructureModuleConfig=StructureModuleConfig())
    layer_norm_s = LayerNorm(cfg.c_s)
    layer_norm_z = LayerNorm(cfg.c_z)
    linear_in = Linear(cfg.c_s => cfg.c_s)
    ipa = ESMFoldIPA(
        cfg.c_s, cfg.c_z, cfg.c_ipa, cfg.no_heads_ipa,
        cfg.no_qk_points, cfg.no_v_points;
        inf=cfg.inf, eps=cfg.epsilon,
    )
    ipa_dropout = SharedDropout(cfg.dropout_rate, [3])
    layer_norm_ipa = LayerNorm(cfg.c_s)
    transition = StructureModuleTransition(cfg.c_s, cfg.no_transition_layers, cfg.dropout_rate)
    bb_update = BackboneUpdate(cfg.c_s)
    angle_resnet = AngleResnet(cfg.c_s, cfg.c_resnet, cfg.no_resnet_blocks, cfg.no_angles, cfg.epsilon)

    return StructureModule(
        cfg, layer_norm_s, layer_norm_z, linear_in,
        ipa, ipa_dropout, layer_norm_ipa, transition,
        bb_update, angle_resnet,
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

    all_preds = Vector{Dict{Symbol,Any}}(undef, m.cfg.no_blocks)
    for i in 1:m.cfg.no_blocks
        s = s .+ m.ipa(s, z, rigids, mask)
        s = m.ipa_dropout(s)
        s = m.layer_norm_ipa(s)
        s = m.transition(s)

        rigids = compose_q_update_vec(rigids, m.bb_update(s))

        backb_to_global = Rigid(RotMatRotation(get_rot_mats(rigids.rots)), rigids.trans)
        backb_to_global = scale_translation(backb_to_global, m.cfg.trans_scale_factor)

        unnormalized_angles, angles = m.angle_resnet(s, s_initial)

        default_frames = to_device(restype_rigid_group_default_frame, angles, eltype(angles))
        group_idx = to_device(restype_atom14_to_rigid_group, angles, Int)
        atom_mask = to_device(restype_atom14_mask, angles, eltype(angles))
        lit_positions = to_device(restype_atom14_rigid_group_positions, angles, eltype(angles))

        all_frames_to_global = torsion_angles_to_frames(backb_to_global, angles, aatype, default_frames)
        pred_xyz = frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global, aatype, default_frames,
            group_idx, atom_mask, lit_positions,
        )

        scaled_rigids = scale_translation(rigids, m.cfg.trans_scale_factor)

        all_preds[i] = Dict{Symbol,Any}(
            :frames => to_tensor_7(scaled_rigids),
            :sidechain_frames => to_tensor_4x4(all_frames_to_global),
            :unnormalized_angles => unnormalized_angles,
            :angles => angles,
            :positions => pred_xyz,
            :states => s,
        )
    end

    out = stack_dicts(all_preds)
    out[:single] = s
    return out
end
