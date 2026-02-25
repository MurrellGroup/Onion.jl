"""
    PairWeightedAveraging(c_m, c_z, c_h, num_heads; inf=1f6)

Pair-weighted averaging of sequence features using pairwise weights.
Input: `m (C_m, S, N, B)`, `z (C_z, N, N, B)`, `mask (N, N, B)`.
Output: `(C_m, S, N, B)`.
"""
@concrete struct PairWeightedAveraging <: Layer
    c_h::Int; num_heads::Int; inf
    norm_m; norm_z; proj_m; proj_g; proj_z; proj_o
end

function PairWeightedAveraging(c_m::Int, c_z::Int, c_h::Int, num_heads::Int; inf::Real=1f6)
    norm_m = LayerNorm(c_m)
    norm_z = LayerNorm(c_z)
    proj_m = Linear(c_m => c_h * num_heads, bias=false)
    proj_g = Linear(c_m => c_h * num_heads, bias=false)
    proj_z = Linear(c_z => num_heads, bias=false)
    proj_o = Linear(c_h * num_heads => c_m, bias=false)
    return PairWeightedAveraging(c_h, num_heads, inf, norm_m, norm_z, proj_m, proj_g, proj_z, proj_o)
end

function (l::PairWeightedAveraging)(m, z, mask)
    c_h, h = l.c_h, l.num_heads

    m = l.norm_m(m)
    z = l.norm_z(z)

    v = l.proj_m(m)
    g = l.proj_g(m)

    s, n, bsz = size(m, 2), size(m, 3), size(m, 4)

    v = permutedims(reshape(v, c_h, h, s, n, bsz), (2, 3, 4, 1, 5))  # (H, S, N, C_h, B)
    g = NNlib.sigmoid.(permutedims(reshape(g, c_h, h, s, n, bsz), (2, 3, 4, 1, 5)))

    # Pair weights: softmax over the N (key) dim
    b = l.proj_z(z)  # (H, N, N, B)
    mask_b = reshape(mask, 1, n, n, bsz)
    b = b .+ (1 .- mask_b) .* (-l.inf)
    w = NNlib.softmax(b; dims=3)

    # Batched aggregation
    w_bat = reshape(permutedims(w, (2, 3, 1, 4)), n, n, h * bsz)
    v_bat = reshape(permutedims(v, (3, 4, 2, 1, 5)), n, c_h * s, h * bsz)
    o_flat = NNlib.batched_mul(w_bat, v_bat)  # (N, C_h*S, H*B)

    o = permutedims(reshape(o_flat, n, c_h, s, h, bsz), (4, 3, 1, 2, 5))  # (H, S, N, C_h, B)
    o = o .* g
    o = reshape(permutedims(o, (4, 1, 2, 3, 5)), c_h * h, s, n, bsz)

    return l.proj_o(o)
end
