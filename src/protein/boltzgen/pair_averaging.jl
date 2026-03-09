using Flux
using NNlib

@concrete struct PairWeightedAveraging <: Layer
    c_m::Int
    c_z::Int
    c_h::Int
    num_heads::Int
    inf
    norm_m
    norm_z
    proj_m
    proj_g
    proj_z
    proj_o
end

@layer PairWeightedAveraging

function PairWeightedAveraging(
    c_m::Int,
    c_z::Int,
    c_h::Int,
    num_heads::Int;
    inf::Real=1f6,
)
    norm_m = BGLayerNorm(c_m; eps=1f-5)
    norm_z = BGLayerNorm(c_z; eps=1f-5)
    proj_m = Dense(c_m => c_h * num_heads, bias=false)
    proj_g = Dense(c_m => c_h * num_heads, bias=false)
    proj_z = Dense(c_z => num_heads, bias=false)
    proj_o = Dense(c_h * num_heads => c_m, bias=false)
    final_init!(proj_o.weight)

    torch_linear_init!(proj_m.weight)
    torch_linear_init!(proj_g.weight)
    torch_linear_init!(proj_z.weight)

    return PairWeightedAveraging(
        c_m,
        c_z,
        c_h,
        num_heads,
        inf,
        norm_m,
        norm_z,
        proj_m,
        proj_g,
        proj_z,
        proj_o,
    )
end

function (layer::PairWeightedAveraging)(m, z, mask; chunk_heads::Bool=false)
    # m: (C_m, S, N, B), z: (C_z, N, N, B), mask: (N, N, B)
    @assert eltype(m) === eltype(layer.proj_m.weight) "PairWeightedAveraging input eltype $(eltype(m)) must match weight eltype $(eltype(layer.proj_m.weight))"
    m = layer.norm_m(m)
    z = layer.norm_z(z)

    v = layer.proj_m(m)
    g = layer.proj_g(m)

    c_h = layer.c_h
    h = layer.num_heads
    s = size(m, 2)
    n = size(m, 3)
    bsz = size(m, 4)

    v = reshape(v, c_h, h, s, n, bsz)
    v = permutedims(v, (2, 3, 4, 1, 5)) # (H, S, N, C_h, B)

    g = reshape(g, c_h, h, s, n, bsz)
    g = permutedims(g, (2, 3, 4, 1, 5))
    g = NNlib.sigmoid.(g)

    b = layer.proj_z(z) # (H, N, N, B)
    mask_b = reshape(mask, 1, n, n, bsz)
    b = b .+ (1 .- mask_b) .* (-layer.inf)
    w = NNlib.softmax(b; dims=3)

    # Compute output: single batched_mul over all S positions at once.
    # w: (H, N, N, B) → (N, N, H*B);  v: (H, S, N, C_h, B) → (N, C_h*S, H*B)
    # Since w is the same for all S, we fold S into the column dimension of v.
    w_perm = permutedims(w, (2, 3, 1, 4)) # (N, N, H, B)
    wbat = reshape(w_perm, n, n, h * bsz)

    v_for_mul = permutedims(v, (3, 4, 2, 1, 5))          # (N, C_h, S, H, B)
    v_for_mul = reshape(v_for_mul, n, c_h * s, h * bsz)  # (N, C_h*S, H*B)

    o_flat = NNlib.batched_mul(wbat, v_for_mul)           # (N, C_h*S, H*B)

    o = reshape(o_flat, n, c_h, s, h, bsz)
    o = permutedims(o, (4, 3, 1, 2, 5))                  # (H, S, N, C_h, B)
    o = o .* g
    o = permutedims(o, (4, 1, 2, 3, 5)) # (C_h, H, S, N, B)
    o = reshape(o, c_h * h, s, n, bsz)

    return layer.proj_o(o)
end

# Original map-loop implementation for parity testing
function _pwa_forward_maploop(layer::PairWeightedAveraging, m, z, mask)
    m = layer.norm_m(m)
    z = layer.norm_z(z)

    v = layer.proj_m(m)
    g = layer.proj_g(m)

    c_h = layer.c_h
    h = layer.num_heads
    s = size(m, 2)
    n = size(m, 3)
    bsz = size(m, 4)

    v = reshape(v, c_h, h, s, n, bsz)
    v = permutedims(v, (2, 3, 4, 1, 5))

    g = reshape(g, c_h, h, s, n, bsz)
    g = permutedims(g, (2, 3, 4, 1, 5))
    g = NNlib.sigmoid.(g)

    b = layer.proj_z(z)
    mask_b = reshape(mask, 1, n, n, bsz)
    b = b .+ (1 .- mask_b) .* (-layer.inf)
    w = NNlib.softmax(b; dims=3)

    w_perm = permutedims(w, (2, 3, 1, 4))
    wbat = reshape(w_perm, n, n, h * bsz)

    outs = map(1:s) do si
        v_s = v[:, si, :, :, :]
        v_perm = permutedims(v_s, (2, 3, 1, 4))
        vbat = reshape(v_perm, n, c_h, h * bsz)

        out_s = NNlib.batched_mul(wbat, vbat)
        out_s = reshape(out_s, n, c_h, h, bsz)
        out_s = permutedims(out_s, (3, 1, 2, 4))
        reshape(out_s, h, 1, n, c_h, bsz)
    end

    o = reduce((a, b) -> cat(a, b; dims=2), outs)
    o = o .* g
    o = permutedims(o, (4, 1, 2, 3, 5))
    o = reshape(o, c_h * h, s, n, bsz)

    return layer.proj_o(o)
end
