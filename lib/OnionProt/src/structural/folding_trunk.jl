# ──── FoldingTrunk (ESMFold) ────

struct FoldingTrunkConfig
    num_blocks::Int
    sequence_state_dim::Int
    pairwise_state_dim::Int
    sequence_head_width::Int
    pairwise_head_width::Int
    position_bins::Int
    dropout::Float32
    max_recycles::Int
    structure_module::StructureModuleConfig
end

function FoldingTrunkConfig()
    return FoldingTrunkConfig(
        48, 1024, 128, 32, 32, 32,
        0f0, 4, StructureModuleConfig(),
    )
end

"""
    RelativePosition(bins, pairwise_state_dim)

Computes pairwise relative position embeddings from residue indices.
"""
@concrete struct RelativePosition <: Layer
    bins::Int
    embedding
end

function RelativePosition(bins::Int, pairwise_state_dim::Int)
    embedding = Embedding(2 * bins + 2 => pairwise_state_dim)
    return RelativePosition(bins, embedding)
end

function (m::RelativePosition)(residue_index::AbstractArray; mask=nothing)
    diff = rearrange(residue_index, einops"L B -> 1 L B") .-
           rearrange(residue_index, einops"L B -> L 1 B")
    diff = clamp.(diff, -m.bins, m.bins)
    diff = diff .+ m.bins .+ 1

    if mask !== nothing
        pair_mask = rearrange(mask, einops"L B -> L 1 B") .* rearrange(mask, einops"L B -> 1 L B")
        diff = ifelse.(pair_mask .== 1, diff, 0)
    end

    diff = diff .+ 1
    return m.embedding(diff)
end

function cross_first(a::AbstractArray, b::AbstractArray)
    a1 = _view_first1(a, 1); a2 = _view_first1(a, 2); a3 = _view_first1(a, 3)
    b1 = _view_first1(b, 1); b2 = _view_first1(b, 2); b3 = _view_first1(b, 3)
    c1 = a2 .* b3 .- a3 .* b2
    c2 = a3 .* b1 .- a1 .* b3
    c3 = a1 .* b2 .- a2 .* b1
    return cat(reshape(c1, 1, size(c1)...), reshape(c2, 1, size(c2)...), reshape(c3, 1, size(c3)...); dims=1)
end

"""
    distogram(coords, min_bin, max_bin, num_bins)

Computes Cb from N/Ca/C via cross product, then bins pairwise Cb-Cb distances.
"""
function distogram(coords, min_bin::Real, max_bin::Real, num_bins::Int)
    boundaries = range(Float32(min_bin), Float32(max_bin); length=num_bins - 1)
    boundaries = collect(boundaries)
    boundaries = to_device(boundaries, coords, Float32) .^ 2

    N = view(coords, :, 1, :, :)
    CA = view(coords, :, 2, :, :)
    C = view(coords, :, 3, :, :)

    b = CA .- N
    c = C .- CA
    a = cross_first(b, c)
    CB = (-0.58273431f0 .* a) .+ (0.56802827f0 .* b) .- (0.54067466f0 .* c) .+ CA

    diff = rearrange(CB, einops"xyz L B -> xyz L 1 B") .-
           rearrange(CB, einops"xyz L B -> xyz 1 L B")
    dists = sum(diff .^ 2; dims=1)
    dists = dropdims(dists; dims=1)

    dists_exp = reshape(dists, size(dists)..., 1)
    bview = reshape(boundaries, 1, 1, 1, :)
    bins = sum(dists_exp .> bview; dims=4)
    bins = dropdims(bins; dims=4)
    return Int.(bins)
end

"""
    FoldingTrunk(; cfg=FoldingTrunkConfig())

ESMFold folding trunk: recycle loop with `TriangularSelfAttentionBlock` stack + `StructureModule`.
"""
@concrete struct FoldingTrunk <: Layer
    cfg::FoldingTrunkConfig
    pairwise_positional_embedding
    blocks
    recycle_bins::Int
    recycle_s_norm; recycle_z_norm; recycle_disto
    structure_module
    trunk2sm_s; trunk2sm_z
end

function FoldingTrunk(; cfg::FoldingTrunkConfig=FoldingTrunkConfig())
    c_s = cfg.sequence_state_dim
    c_z = cfg.pairwise_state_dim

    pairwise_positional_embedding = RelativePosition(cfg.position_bins, c_z)
    blocks = [
        TriangularSelfAttentionBlock(
            c_s, c_z, cfg.sequence_head_width, cfg.pairwise_head_width;
            dropout=cfg.dropout,
        )
        for _ in 1:cfg.num_blocks
    ]

    recycle_bins = 15
    recycle_s_norm = LayerNorm(c_s)
    recycle_z_norm = LayerNorm(c_z)
    recycle_disto = Embedding(recycle_bins => c_z)
    recycle_disto.weight[:, 1] .= 0f0

    sm_cfg = cfg.structure_module
    structure_module = StructureModule(cfg=sm_cfg)
    trunk2sm_s = Linear(c_s => sm_cfg.c_s)
    trunk2sm_z = Linear(c_z => sm_cfg.c_z)

    return FoldingTrunk(
        cfg, pairwise_positional_embedding, blocks, recycle_bins,
        recycle_s_norm, recycle_z_norm, recycle_disto,
        structure_module, trunk2sm_s, trunk2sm_z,
    )
end

function _trunk_iter(m::FoldingTrunk, s, z, residx, mask)
    z = z .+ m.pairwise_positional_embedding(residx; mask=mask)
    block_mask = mask
    if mask !== nothing
        mask_total = sum(mask)
        mask_sum = mask_total isa AbstractArray ? Int(Array(mask_total)[]) : Int(mask_total)
        if mask_sum == length(mask)
            block_mask = nothing
        end
    end
    for block in m.blocks
        s, z = block(s, z; mask=block_mask)
    end
    return s, z
end

function (m::FoldingTrunk)(
    seq_feats, pair_feats, true_aa, residx, mask;
    no_recycles=nothing,
)
    s_s_0 = seq_feats
    s_z_0 = pair_feats

    if no_recycles === nothing
        no_recycles = m.cfg.max_recycles
    else
        no_recycles += 1
    end
    no_recycles < 1 && (no_recycles = 1)

    recycle_s = zeros_like(s_s_0, size(s_s_0)...)
    recycle_z = zeros_like(s_z_0, size(s_z_0)...)
    recycle_bins = zeros_like(s_z_0, Int, size(s_z_0, 2), size(s_z_0, 3), size(s_z_0, 4))

    mask_f = mask === nothing ? nothing : convert.(eltype(s_s_0), mask)

    structure = nothing
    for _ in 1:no_recycles
        recycle_s = m.recycle_s_norm(recycle_s)
        recycle_z = m.recycle_z_norm(recycle_z)
        recycle_z = recycle_z .+ m.recycle_disto(recycle_bins .+ 1)

        s_s, s_z = _trunk_iter(m, s_s_0 .+ recycle_s, s_z_0 .+ recycle_z, residx, mask)

        structure = m.structure_module(
            Dict(:single => m.trunk2sm_s(s_s), :pair => m.trunk2sm_z(s_z)),
            true_aa, mask_f,
        )

        recycle_s = s_s
        recycle_z = s_z

        coords = structure[:positions][end, :, 1:3, :, :]
        recycle_bins = @ignore_derivatives distogram(coords, 3.375f0, 21.375f0, m.recycle_bins)
    end

    structure === nothing && error("FoldingTrunk: no recycle iterations ran.")
    structure[:s_s] = s_s
    structure[:s_z] = s_z
    return structure
end
