# Protein structure prediction layers
# CPU-only implementations with dispatch hooks for GPU acceleration via OnionTile.

using ChainRulesCore: @ignore_derivatives

# Device transfer utility (works for CPU arrays; CUDA dispatch added by OnionTile)
function to_device(x::AbstractArray, like::AbstractArray, ::Type{T}=eltype(x)) where {T}
    return @ignore_derivatives begin
        y = similar(like, T, size(x))
        copyto!(y, T.(x))
        y
    end
end

function to_device(x::Number, like::AbstractArray, ::Type{T}=typeof(x)) where {T}
    return @ignore_derivatives T(x)
end

include("rigid.jl")
include("residue_constants.jl")
include("openfold_utils.jl")
include("layernorm.jl")
include("openfold_feats.jl")
include("rotary.jl")
include("esmfold_misc.jl")
include("attention.jl")
include("triangular.jl")
include("structure_module.jl")
include("esmfold_embed.jl")
include("folding_trunk.jl")

# Dispatch hooks (overridden by OnionTile for GPU)
export layernorm_first_forward, flash_attention_forward, flash_attention_bias_forward
export rotary_pos_emb_forward, combine_projections_forward

# Rigid body types and utilities
export AbstractRotation, RotMatRotation, QuatRotation, Rigid
export rot_from_mat, rot_from_quat, rot_matmul_first, rot_vec_mul_first
export quat_multiply_first, quat_multiply_by_vec_first, quat_to_rot_first
export rotation_identity, get_rot_mats, get_quats
export compose_q_update_vec, rigid_identity, apply_rotation, apply_rigid
export invert_apply_rigid, compose, scale_translation, to_tensor_7, to_tensor_4x4
export rigid_index

# Utility functions
export permute_final_dims, flatten_final_dims, dict_multimap, stack_dicts
export one_hot_last, collate_dense_tensors

# OpenFold feature functions
export rigid_from_tensor_4x4, torsion_angles_to_frames
export frames_and_literature_positions_to_atom14_pos, atom14_to_atom37

# Residue constants
export restypes, restypes_with_x, restype_order, restype_order_with_x, restype_num
export atom_types, atom_order, atom_type_num
export restype_1to3, restype_3to1
export residue_atoms, restype_name_to_atom14_names
export restype_atom14_to_rigid_group, restype_atom14_mask
export restype_atom14_rigid_group_positions, restype_rigid_group_default_frame

# LayerNorm and Linear
export LayerNormFirst, LinearFirst, layernorm_inplace!

# Rotary embeddings
export RotaryEmbedding, rotate_half, apply_rotary_pos_emb

# ESMFold misc
export ESMFoldAttention, SharedDropout, SequenceToPair, PairToSequence, ResidueMLP
export set_training!, is_training
export encode_sequence, batch_encode_sequences
export CategoricalMixture, categorical_lddt

# Attention
export ESMMultiheadAttention

# Triangular
export OFMultiheadAttention, TriangleAttention, TriangleMultiplicativeUpdate
export TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming
export TriangularSelfAttentionBlock

# Structure module
export StructureModuleConfig, PointProjection, ESMFoldIPA
export BackboneUpdate, StructureModuleTransitionLayer, StructureModuleTransition
export AngleResnetBlock, AngleResnet, StructureModule

# Embedding
export ESMFoldEmbedConfig, LayerNormMLP

# Folding trunk
export FoldingTrunkConfig, RelativePosition, FoldingTrunk
export cross_first, distogram, set_chunk_size!
