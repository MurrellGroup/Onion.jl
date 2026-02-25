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
    mask4 = reshape(mask, 1, size(mask)...)
    a = l.proj_a(m) .* mask4  # (C_h, S, N, B)
    b = l.proj_b(m) .* mask4

    s, n, bsz = size(m, 2), size(m, 3), size(m, 4)

    # Mask sum for normalization: (N, N, B)
    mask_t = permutedims(mask, (2, 1, 3))  # (N, S, B)
    mask_sum = max.(NNlib.batched_mul(mask_t, mask), one(T))

    # Outer product via batched_mul contracting over S
    a_flat = reshape(permutedims(a, (1, 3, 2, 4)), c_h * n, s, bsz)
    b_flat = reshape(permutedims(b, (2, 1, 3, 4)), s, c_h * n, bsz)
    z_flat = NNlib.batched_mul(a_flat, b_flat)  # (C_h*N, C_h*N, B)

    z_5d = reshape(z_flat, c_h, n, c_h, n, bsz)
    z = reshape(permutedims(z_5d, (3, 1, 2, 4, 5)), c_h * c_h, n, n, bsz)
    z = z ./ reshape(mask_sum, 1, n, n, bsz)

    return l.proj_o(z)
end
