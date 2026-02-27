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

    v_raw = l.proj_m(m)
    g_raw = l.proj_g(m)

    s, n, bsz = size(m, 2), size(m, 3), size(m, 4)

    v = rearrange(v_raw, einops"(Ch H) S N B -> H S N Ch B"; Ch=c_h)
    g = NNlib.sigmoid.(rearrange(g_raw, einops"(Ch H) S N B -> H S N Ch B"; Ch=c_h))

    # Pair weights: softmax over the N (key) dim
    b = l.proj_z(z)  # (H, Nq, Nk, B)
    mask_b = rearrange(mask, einops"Nq Nk B -> 1 Nq Nk B")
    b = b .+ (1 .- mask_b) .* (-l.inf)
    w = NNlib.softmax(b; dims=3)

    # Batched aggregation
    w_bat = rearrange(w, einops"H Nq Nk B -> Nq Nk (H B)")
    v_bat = rearrange(v, einops"H S N Ch B -> N (Ch S) (H B)")
    o_flat = NNlib.batched_mul(w_bat, v_bat)  # (N, Ch*S, H*B)

    o = rearrange(o_flat, einops"N (Ch S) (H B) -> H S N Ch B"; Ch=c_h, S=s, H=h)
    o = o .* g
    o = rearrange(o, einops"H S N Ch B -> (Ch H) S N B")

    return l.proj_o(o)
end
