"""
    OuterProductMean(c_in, c_hidden, c_out)

Outer product mean over the MSA sequence dimension.
Input: `m (C_in, S, N, B)`, `mask (S, N, B)`.
Output: `(C_out, N, N, B)`.
"""
@concrete struct OuterProductMean <: Layer
    c_hidden::Int
    norm; proj_a; proj_b; proj_o
end

function OuterProductMean(c_in::Int, c_hidden::Int, c_out::Int)
    norm   = LayerNorm(c_in)
    proj_a = Linear(c_in => c_hidden, bias=false)
    proj_b = Linear(c_in => c_hidden, bias=false)
    proj_o = Linear(c_hidden * c_hidden => c_out)
    return OuterProductMean(c_hidden, norm, proj_a, proj_b, proj_o)
end

function (l::OuterProductMean)(m, mask)
    T = eltype(m)
    c_h = l.c_hidden

    m = l.norm(m)
    mask4 = rearrange(mask, einops"S N B -> 1 S N B")
    a = l.proj_a(m) .* mask4  # (Ch, S, N, B)
    b = l.proj_b(m) .* mask4

    n, bsz = size(m, 3), size(m, 4)

    # Mask sum for normalization: (N, N, B)
    mask_t = rearrange(mask, einops"S N B -> N S B")
    mask_sum = max.(NNlib.batched_mul(mask_t, mask), one(T))

    # Outer product via batched_mul contracting over S
    a_flat = rearrange(a, einops"Ch S N B -> (Ch N) S B")
    b_flat = rearrange(b, einops"Ch S N B -> S (Ch N) B")
    z_flat = NNlib.batched_mul(a_flat, b_flat)  # (Ch*N, Ch*N, B)

    z = rearrange(reshape(z_flat, c_h, n, c_h, n, bsz), einops"Ch1 N1 Ch2 N2 B -> (Ch2 Ch1) N1 N2 B")
    z = z ./ rearrange(mask_sum, einops"N1 N2 B -> 1 N1 N2 B")

    return l.proj_o(z)
end
