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
    mask = eltype(m).(mask)

    m = layer.norm(m)
    a = layer.proj_a(m) .* reshape(mask, 1, size(mask, 1), size(mask, 2), size(mask, 3))
    b = layer.proj_b(m) .* reshape(mask, 1, size(mask, 1), size(mask, 2), size(mask, 3))

    c_hidden = layer.c_hidden
    s = size(m, 2)
    n = size(m, 3)
    bsz = size(m, 4)

    af = Float32.(a)
    bf = Float32.(b)

    # Compute mask sum: (N, N, B)
    mask_f = Float32.(mask)
    mask1 = permutedims(mask_f, (2, 1, 3)) # (N, S, B)
    mask_sum = NNlib.batched_mul(mask1, mask_f) # (N, N, B)
    mask_sum = max.(mask_sum, 1f0)

    # Compute outer product mean (unchunked)
    z = mapreduce(+, 1:s) do si
        a_s = af[:, si, :, :] # (C_h, N, B)
        b_s = bf[:, si, :, :] # (C_h, N, B)
        reshape(a_s, c_hidden, 1, n, 1, bsz) .* reshape(b_s, 1, c_hidden, 1, n, bsz)
    end

    z = permutedims(z, (2, 1, 3, 4, 5)) # (C_h, C_h) -> (d, c) for Python-style flatten
    z = reshape(z, c_hidden * c_hidden, n, n, bsz)
    z = z ./ reshape(mask_sum, 1, n, n, bsz)
    z = eltype(m).(z)

    return layer.proj_o(z)
end
