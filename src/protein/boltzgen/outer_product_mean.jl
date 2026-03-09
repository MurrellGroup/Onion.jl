using Flux
using NNlib

@concrete struct OuterProductMean <: Layer
    c_hidden::Int
    norm
    proj_a
    proj_b
    proj_o
end

@layer OuterProductMean

function OuterProductMean(c_in::Int, c_hidden::Int, c_out::Int)
    norm = BGLayerNorm(c_in; eps=1f-5)
    proj_a = Dense(c_in => c_hidden, bias=false)
    proj_b = Dense(c_in => c_hidden, bias=false)
    proj_o = Dense(c_hidden * c_hidden => c_out)

    final_init!(proj_o.weight)
    final_init!(proj_o.bias)

    torch_linear_init!(proj_a.weight)
    torch_linear_init!(proj_b.weight)

    return OuterProductMean(c_hidden, norm, proj_a, proj_b, proj_o)
end

function (layer::OuterProductMean)(m, mask; chunk_size::Union{Nothing,Int}=nothing)
    # m: (C_in, S, N, B)
    @assert eltype(m) === eltype(layer.proj_a.weight) "OuterProductMean input eltype $(eltype(m)) must match weight eltype $(eltype(layer.proj_a.weight))"
    @assert eltype(mask) === eltype(m) "OuterProductMean mask eltype $(eltype(mask)) must match input eltype $(eltype(m))"

    m = layer.norm(m)
    a = layer.proj_a(m) .* reshape(mask, 1, size(mask, 1), size(mask, 2), size(mask, 3))
    b = layer.proj_b(m) .* reshape(mask, 1, size(mask, 1), size(mask, 2), size(mask, 3))

    c_hidden = layer.c_hidden
    s = size(m, 2)
    n = size(m, 3)
    bsz = size(m, 4)

    # Compute mask sum: (N, N, B)
    mask1 = permutedims(mask, (2, 1, 3)) # (N, S, B)
    mask_sum = NNlib.batched_mul(mask1, mask) # (N, N, B)
    mask_sum = max.(mask_sum, eltype(m)(1))

    # Outer product mean via batched_mul (contracts over S in one BLAS call)
    # z[c1, c2, n1, n2, b] = Σ_s a[c1, s, n1, b] * b[c2, s, n2, b]
    # Flatten (C_h, N) into one dim, contract over S via batched_mul
    a_flat = reshape(permutedims(a, (1, 3, 2, 4)), c_hidden * n, s, bsz)  # (C_h*N, S, B)
    b_flat = reshape(permutedims(b, (2, 1, 3, 4)), s, c_hidden * n, bsz)  # (S, C_h*N, B)
    z_flat = NNlib.batched_mul(a_flat, b_flat)  # (C_h*N, C_h*N, B)

    # Reshape to (C_h, N, C_h, N, B) then rearrange to (C_h*C_h, N, N, B)
    z_5d = reshape(z_flat, c_hidden, n, c_hidden, n, bsz)
    z = reshape(permutedims(z_5d, (3, 1, 2, 4, 5)), c_hidden * c_hidden, n, n, bsz)
    z = z ./ reshape(mask_sum, 1, n, n, bsz)

    return layer.proj_o(z)
end

# Original mapreduce implementation for parity testing
function _opm_forward_mapreduce(layer::OuterProductMean, m, mask)
    mask = eltype(m).(mask)

    m = layer.norm(m)
    a = layer.proj_a(m) .* reshape(mask, 1, size(mask, 1), size(mask, 2), size(mask, 3))
    b = layer.proj_b(m) .* reshape(mask, 1, size(mask, 1), size(mask, 2), size(mask, 3))

    c_hidden = layer.c_hidden
    s = size(m, 2)
    n = size(m, 3)
    bsz = size(m, 4)

    mask1 = permutedims(mask, (2, 1, 3))
    mask_sum = NNlib.batched_mul(mask1, mask)
    mask_sum = max.(mask_sum, eltype(m)(1))

    z = mapreduce(+, 1:s) do si
        a_s = a[:, si, :, :]
        b_s = b[:, si, :, :]
        reshape(a_s, c_hidden, 1, n, 1, bsz) .* reshape(b_s, 1, c_hidden, 1, n, bsz)
    end

    z = permutedims(z, (2, 1, 3, 4, 5))
    z = reshape(z, c_hidden * c_hidden, n, n, bsz)
    z = z ./ reshape(mask_sum, 1, n, n, bsz)

    return layer.proj_o(z)
end
