using NNlib

const BG_CHAIN_TYPE_IDS = Dict(
    "PROTEIN" => 0,
    "DNA" => 1,
    "RNA" => 2,
    "NONPOLYMER" => 3,
)

function _unsqueeze(x, dim::Int)
    sz = collect(size(x))
    insert!(sz, dim, 1)
    return reshape(x, Tuple(sz))
end

_max_drop(x; dims::Int) = dropdims(maximum(x; dims=dims); dims=dims)

function compute_collinear_mask(v1, v2)
    norm1 = sqrt.(sum(abs2, v1; dims=2))
    norm2 = sqrt.(sum(abs2, v2; dims=2))
    v1n = v1 ./ (norm1 .+ 1f-6)
    v2n = v2 ./ (norm2 .+ 1f-6)
    mask_angle = abs.(sum(v1n .* v2n; dims=2)) .< 0.9063f0
    mask_overlap1 = vec(norm1) .> 1f-2
    mask_overlap2 = vec(norm2) .> 1f-2
    return mask_angle .& mask_overlap1 .& mask_overlap2
end

function _repeat_interleave(x, repeats::Int, dim::Int)
    repeats == 1 && return x
    idx = repeat(collect(1:size(x, dim)), inner=repeats)
    inds = ntuple(i -> i == dim ? idx : Colon(), ndims(x))
    return x[inds...]
end

function compute_frame_pred(
    pred_atom_coords,
    frames_idx_true,
    feats,
    multiplicity;
    resolved_mask=nothing,
    inference::Bool=false,
)
    asym_id_token = permutedims(feats["asym_id"], (2, 1))
    atom_to_token = permutedims(feats["atom_to_token"], (3, 1, 2))
    token_pad_mask = permutedims(feats["token_pad_mask"], (2, 1))
    atom_pad_mask = permutedims(feats["atom_pad_mask"], (2, 1))
    mol_type = permutedims(feats["mol_type"], (2, 1))
    frames_idx_true = permutedims(frames_idx_true, (3, 1, 2))
    # Feature tensors store atom indices in 0-based convention; convert to Julia 1-based.
    if !isempty(frames_idx_true) && minimum(frames_idx_true) == 0
        frames_idx_true = frames_idx_true .+ 1
    end

    # batched matmul to map atom->token ids
    b = size(atom_to_token, 1)
    n_atom = size(atom_to_token, 2)
    n_token = size(atom_to_token, 3)
    asym_id_atom = Array{Float32}(undef, b, n_atom)
    for bi in 1:b
        asym_id_atom[bi, :] = (atom_to_token[bi, :, :] * reshape(Float32.(asym_id_token[bi, :]), n_token, 1))[:, 1]
    end

    B, N, _ = size(pred_atom_coords)
    pred_atom_coords = reshape(pred_atom_coords, div(B, multiplicity), multiplicity, :, 3)

    frames_idx_pred = _repeat_interleave(frames_idx_true, multiplicity, 1)
    frames_idx_pred = reshape(frames_idx_pred, div(B, multiplicity), multiplicity, :, 3)

    for i in 1:size(pred_atom_coords, 1)
        token_idx = 1
        atom_idx = 1
        for id in unique(asym_id_token[i, :])
            mask_chain_token = (asym_id_token[i, :] .== id) .* token_pad_mask[i, :]
            mask_chain_atom = (asym_id_atom[i, :] .== id) .* atom_pad_mask[i, :]
            num_tokens = Int(sum(mask_chain_token))
            num_atoms = Int(sum(mask_chain_atom))

            if mol_type[i, token_idx] != BG_CHAIN_TYPE_IDS["NONPOLYMER"] || num_atoms < 3
                token_idx += num_tokens
                atom_idx += num_atoms
                continue
            end

            mask_chain_atom_bool = mask_chain_atom .> 0
            coords = pred_atom_coords[i, :, mask_chain_atom_bool, :]
            # coords: (multiplicity, num_atoms, 3)
            dist_mat = zeros(Float32, multiplicity, num_atoms, num_atoms)
            for m in 1:multiplicity
                for a in 1:num_atoms
                    for b2 in 1:num_atoms
                        d = coords[m, a, :] .- coords[m, b2, :]
                        dist_mat[m, a, b2] = sqrt(sum(abs2, d))
                    end
                end
            end

            if inference
                resolved_pair = 1 .- (atom_pad_mask[i, mask_chain_atom_bool]') * (atom_pad_mask[i, mask_chain_atom_bool])
            else
                if resolved_mask === nothing
                    resolved_mask = permutedims(feats["atom_resolved_mask"], (2, 1))
                end
                resolved_pair = 1 .- (resolved_mask[i, mask_chain_atom_bool]') * (resolved_mask[i, mask_chain_atom_bool])
            end
            resolved_pair = Float32.(resolved_pair)
            resolved_pair[resolved_pair .== 1f0] .= Inf

            indices = Array{Int}(undef, multiplicity, num_atoms, num_atoms)
            for m in 1:multiplicity
                for a in 1:num_atoms
                    idx = sortperm(dist_mat[m, a, :] .+ resolved_pair[a, :])
                    indices[m, a, :] = idx
                end
            end

            frames = cat(
                indices[:, :, 2:2],
                indices[:, :, 1:1],
                indices[:, :, 3:3];
                dims=3,
            ) .+ (atom_idx - 1)

            frames_idx_pred[i, :, token_idx:token_idx+num_atoms-1, :] = frames

            token_idx += num_tokens
            atom_idx += num_atoms
        end
    end

    # Expand frames
    frames_expanded = Array{eltype(pred_atom_coords)}(undef, size(pred_atom_coords, 1), size(pred_atom_coords, 2), size(frames_idx_pred, 3), 3, 3)
    for bi in 1:size(pred_atom_coords, 1)
        for mi in 1:size(pred_atom_coords, 2)
            for ti in 1:size(frames_idx_pred, 3)
                for fi in 1:3
                    idx = frames_idx_pred[bi, mi, ti, fi]
                    frames_expanded[bi, mi, ti, fi, :] = pred_atom_coords[bi, mi, idx, :]
                end
            end
        end
    end

    frames_expanded = reshape(frames_expanded, :, 3, 3)

    mask_collinear_pred = compute_collinear_mask(
        frames_expanded[:, 2, :] .- frames_expanded[:, 1, :],
        frames_expanded[:, 2, :] .- frames_expanded[:, 3, :],
    )
    mask_collinear_pred = reshape(mask_collinear_pred, size(pred_atom_coords, 1), size(pred_atom_coords, 2), :)

    return frames_idx_pred, mask_collinear_pred .* reshape(token_pad_mask, size(pred_atom_coords, 1), 1, :)
end

function compute_aggregated_metric(logits; end_value=1.0)
    num_bins = size(logits, ndims(logits))
    bin_width = end_value / num_bins
    T = eltype(logits)
    # Bounds are fixed constants — wrap allocation+transfer in @ignore_derivatives
    bounds = ChainRulesCore.@ignore_derivatives begin
        bounds_cpu = T.(collect(range(0.5 * bin_width, step=bin_width, length=num_bins)))
        copyto!(similar(logits, num_bins), bounds_cpu)
    end
    probs = NNlib.softmax(logits; dims=ndims(logits))
    shape_prefix = ntuple(_ -> 1, ndims(probs) - 1)
    bounds_view = reshape(bounds, shape_prefix..., num_bins)
    return dropdims(sum(probs .* bounds_view; dims=ndims(probs)); dims=ndims(probs))
end

function tm_function(d, Nres)
    d0 = 1.24f0 * (max.(Nres, 19f0) .- 15f0) .^ (1f0 / 3f0) .- 1.8f0
    return 1f0 ./ (1f0 .+ (d ./ d0) .^ 2)
end

function compute_ptms(logits, x_preds, feats, multiplicity)
    _, mask_collinear_pred = compute_frame_pred(x_preds, feats["frames_idx"], feats, multiplicity; inference=true)

    token_pad_mask = permutedims(feats["token_pad_mask"], (2, 1))
    asym_id = permutedims(feats["asym_id"], (2, 1))
    mol_type = permutedims(feats["mol_type"], (2, 1))
    design_mask = permutedims(feats["design_mask"], (2, 1))
    chain_design_mask = permutedims(feats["chain_design_mask"], (2, 1))
    atom_to_token = permutedims(feats["atom_to_token"], (3, 1, 2))
    atom_pad_mask = permutedims(feats["atom_pad_mask"], (2, 1))

    mask_pad = _repeat_interleave(token_pad_mask, multiplicity, 1)
    maski = reshape(mask_collinear_pred, :, size(mask_collinear_pred, 3))
    pair_mask_ptm = _unsqueeze(maski, 3) .* _unsqueeze(mask_pad, 2) .* _unsqueeze(mask_pad, 3)

    asym_id = _repeat_interleave(asym_id, multiplicity, 1)
    pair_mask_iptm = _unsqueeze(maski, 3) .* (_unsqueeze(asym_id, 2) .!= _unsqueeze(asym_id, 3)) .* _unsqueeze(mask_pad, 2) .* _unsqueeze(mask_pad, 3)

    num_bins = size(logits, ndims(logits))
    bin_width = 32.0f0 / num_bins
    pae_value = collect(range(0.5f0 * bin_width, step=bin_width, length=num_bins))
    N_res = sum(mask_pad; dims=2)
    tm_value = tm_function(reshape(pae_value, 1, 1, 1, :), N_res) # broadcast

    probs = NNlib.softmax(logits; dims=ndims(logits))
    tm_expected_value = sum(probs .* tm_value; dims=ndims(probs))

    ptm = _max_drop(sum(tm_expected_value .* pair_mask_ptm; dims=3) ./ (sum(pair_mask_ptm; dims=3) .+ 1f-5); dims=2)
    iptm = _max_drop(sum(tm_expected_value .* pair_mask_iptm; dims=3) ./ (sum(pair_mask_iptm; dims=3) .+ 1f-5); dims=2)

    token_type = _repeat_interleave(mol_type, multiplicity, 1)
    is_ligand_token = token_type .== BG_CHAIN_TYPE_IDS["NONPOLYMER"]
    is_protein_token = token_type .== BG_CHAIN_TYPE_IDS["PROTEIN"]

    ligand_iptm_mask = _unsqueeze(maski, 3) .* (_unsqueeze(asym_id, 2) .!= _unsqueeze(asym_id, 3)) .* _unsqueeze(mask_pad, 2) .* _unsqueeze(mask_pad, 3) .* (
        _unsqueeze(is_ligand_token, 3) .* _unsqueeze(is_protein_token, 2) .+ _unsqueeze(is_protein_token, 3) .* _unsqueeze(is_ligand_token, 2)
    )

    protein_iptm_mask = _unsqueeze(maski, 3) .* (_unsqueeze(asym_id, 2) .!= _unsqueeze(asym_id, 3)) .* _unsqueeze(mask_pad, 2) .* _unsqueeze(mask_pad, 3) .* (
        _unsqueeze(is_protein_token, 3) .* _unsqueeze(is_protein_token, 2)
    )

    ligand_iptm = _max_drop(sum(tm_expected_value .* ligand_iptm_mask; dims=3) ./ (sum(ligand_iptm_mask; dims=3) .+ 1f-5); dims=2)
    protein_iptm = _max_drop(sum(tm_expected_value .* protein_iptm_mask; dims=3) ./ (sum(protein_iptm_mask; dims=3) .+ 1f-5); dims=2)

    if sum(design_mask) > 0
        is_chain_design_token = chain_design_mask
        is_target_token = 1 .- chain_design_mask
    else
        is_chain_design_token = design_mask
        is_target_token = 1 .- design_mask
    end

    design_iptm_mask = _unsqueeze(maski, 3) .* _unsqueeze(mask_pad, 2) .* _unsqueeze(mask_pad, 3) .* (
        _unsqueeze(is_chain_design_token, 3) .* _unsqueeze(is_target_token, 2) .+ _unsqueeze(is_target_token, 3) .* _unsqueeze(is_chain_design_token, 2)
    )

    design_iptm = _max_drop(sum(tm_expected_value .* design_iptm_mask; dims=3) ./ (sum(design_iptm_mask; dims=3) .+ 1f-5); dims=2)

    is_design_token = design_mask
    is_target_token = 1 .- is_chain_design_token

    design_to_target_iptm_mask = _unsqueeze(maski, 3) .* _unsqueeze(mask_pad, 2) .* _unsqueeze(mask_pad, 3) .* (
        _unsqueeze(is_design_token, 3) .* _unsqueeze(is_target_token, 2) .+ _unsqueeze(is_target_token, 3) .* _unsqueeze(is_design_token, 2)
    )

    design_to_target_iptm = _max_drop(sum(tm_expected_value .* design_to_target_iptm_mask; dims=3) ./ (sum(design_to_target_iptm_mask; dims=3) .+ 1f-5); dims=2)

    design_ptm_mask = _unsqueeze(maski, 3) .* _unsqueeze(mask_pad, 2) .* _unsqueeze(mask_pad, 3) .* (
        _unsqueeze(is_chain_design_token, 3) .* _unsqueeze(is_chain_design_token, 2)
    )

    target_ptm_mask = _unsqueeze(maski, 3) .* _unsqueeze(mask_pad, 2) .* _unsqueeze(mask_pad, 3) .* (
        _unsqueeze(is_target_token, 3) .* _unsqueeze(is_target_token, 2)
    )

    design_ptm = _max_drop(sum(tm_expected_value .* design_ptm_mask; dims=3) ./ (sum(design_ptm_mask; dims=3) .+ 1f-5); dims=2)
    target_ptm = _max_drop(sum(tm_expected_value .* target_ptm_mask; dims=3) ./ (sum(target_ptm_mask; dims=3) .+ 1f-5); dims=2)

    chain_pair_iptm = Dict{Int,Dict{Int,Any}}()
    for idx1 in unique(asym_id)
        chain_iptm = Dict{Int,Any}()
        for idx2 in unique(asym_id)
            mask_pair_chain = _unsqueeze(maski, 3) .* (_unsqueeze(asym_id, 2) .== idx1) .* (_unsqueeze(asym_id, 3) .== idx2) .* _unsqueeze(mask_pad, 2) .* _unsqueeze(mask_pad, 3)
            chain_iptm[idx2] = _max_drop(sum(tm_expected_value .* mask_pair_chain; dims=3) ./ (sum(mask_pair_chain; dims=3) .+ 1f-5); dims=2)
        end
        chain_pair_iptm[idx1] = chain_iptm
    end

    # Interface interaction pTM
    x_pred_half = Float16.(x_preds)
    interact_mask = zeros(eltype(design_mask), size(design_mask))

    if sum(is_chain_design_token) > 0 && sum(1 .- is_chain_design_token) > 0
        B = size(atom_to_token, 1)
        N_atom = size(atom_to_token, 2)
        N_token = size(atom_to_token, 3)

        atom_chain_design_mask = falses(B, N_atom)
        for bi in 1:B
            atom_chain_design_mask[bi, :] = (atom_to_token[bi, :, :] * reshape(Float16.(chain_design_mask[bi, :]), N_token, 1))[:, 1] .> 0
        end

        atom_interact_mask = falses(B, N_atom)
        for bi in 1:B
            design_mask_b = atom_chain_design_mask[bi, :]
            target_mask_b = .!design_mask_b
            if sum(design_mask_b) == 0 || sum(target_mask_b) == 0
                continue
            end
            x_target = x_pred_half[bi, target_mask_b, :]
            x_design = x_pred_half[bi, design_mask_b, :]
            dists = zeros(Float32, size(x_target, 1), size(x_design, 1))
            for i in 1:size(x_target, 1)
                for j in 1:size(x_design, 1)
                    d = x_target[i, :] .- x_design[j, :]
                    dists[i, j] = sqrt(sum(abs2, d))
                end
            end
            min_dists = vec(minimum(dists; dims=1))
            atom_interact_mask[bi, design_mask_b] .= min_dists .< 8f0
        end

        index = zeros(Int, size(atom_to_token, 1), size(atom_to_token, 2))
        for bi in 1:size(atom_to_token, 1)
            for ai in 1:size(atom_to_token, 2)
                tok = findfirst(>(0), atom_to_token[bi, ai, :])
                if tok !== nothing
                    index[bi, ai] = tok
                end
            end
        end
        for bi in 1:size(index, 1)
            for ai in 1:size(index, 2)
                tok = index[bi, ai]
                if tok > 0
                    interact_mask[bi, tok] = interact_mask[bi, tok] | atom_interact_mask[bi, ai]
                end
            end
        end
    end

    is_interacting = interact_mask
    is_target_token = 1 .- is_chain_design_token
    design_iiptm_mask = _unsqueeze(maski, 3) .* _unsqueeze(mask_pad, 2) .* _unsqueeze(mask_pad, 3) .* (
        _unsqueeze(is_interacting, 3) .* _unsqueeze(is_target_token, 2) .+ _unsqueeze(is_target_token, 3) .* _unsqueeze(is_interacting, 2)
    )

    design_iiptm = _max_drop(sum(tm_expected_value .* design_iiptm_mask; dims=3) ./ (sum(design_iiptm_mask; dims=3) .+ 1f-5); dims=2)

    return (
        ptm,
        iptm,
        ligand_iptm,
        protein_iptm,
        chain_pair_iptm,
        design_to_target_iptm,
        design_iptm,
        design_iiptm,
        target_ptm,
        design_ptm,
    )
end

@concrete struct GaussianSmearing <: Layer
    num_gaussians::Int
    coeff
    offset
end

@layer GaussianSmearing

function GaussianSmearing(; start::Float32=0f0, stop::Float32=5f0, num_gaussians::Int=50)
    offset = collect(range(start, stop; length=num_gaussians))
    coeff = -0.5f0 / (offset[2] - offset[1])^2
    return GaussianSmearing(num_gaussians, coeff, offset)
end

function (gs::GaussianSmearing)(dist)
    shape = size(dist)
    dist = reshape(dist, :, 1) .- reshape(gs.offset, 1, :)
    out = exp.(gs.coeff .* dist .^ 2)
    return reshape(out, shape..., gs.num_gaussians)
end
