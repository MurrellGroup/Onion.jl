"""
    FSQ(l, chunk_size)

Finite Scalar Quantization.
`l` is the number of quantization levels. For a sequence with `d` channels, the codebook size would be `l^d`.
`chunk_size` is the number of channels that get combined/separated when `chunk`/`unchunk` are called.
"""
struct FSQ{A<:Integer,B} <: Layer
    l::A
    q::B
    chunk_size::Int
end

function FSQ(l::A, chunk_size::Int) where A
    scal = Int(floor.(l/2))
    f(z::AbstractArray{T}) where T = scal .* Flux.tanh.(z)
    return FSQ(l, f, chunk_size)
end

diffround(x) = x .+ ignore_derivatives(round.(x) .- x)

(fsq::FSQ)(x) = diffround(fsq.q(x))

"""
    chunk(x, q::FSQ, chunk_size)

Make a long quantized sequence shorter and wider (to make it more transformer-friendly).
`x` may have a batch dimension. Contiguous chunks of `chunk_size` are recoded as a single integer in the product space `q.l^chunk_size``.
"""
function chunk(x, q::FSQ)
    s = size(x)
    v = similar(x, q.chunk_size)
    v .= q.l .^ (0:(q.chunk_size-1))
    lb = -floor(Int, q.l/2)
    return 1 .+ Int.(reshape(sum(reshape((x .- lb), q.chunk_size, Int(s[1]/q.chunk_size), s[2:end]...) .* v, dims=1), Int(s[1]/q.chunk_size), s[2:end]...))
end

"""
    unchunk(x, q::FSQ)

Take a sequence that has been `chunk`ed, and expand it back to the original length.
`x == unchunk(chunk(x,q),q)` should be true.
"""
function unchunk(x, q::FSQ)
    idx = x .- 1
    lb = -floor(Int, q.l / 2)
    powers_vec = similar(idx, q.chunk_size)
    powers_vec .= q.l .^ (0:(q.chunk_size-1))
    powers = reshape(powers_vec, (1, q.chunk_size, ntuple(_ -> 1, max(0, ndims(idx) - 1))...))
    centered_digits = rem.(div.(reshape(idx, (size(idx, 1), 1, size(idx)[2:end]...)), powers), q.l) .+ lb
    perm = (2, 1, 3:ndims(centered_digits)...)
    return reshape(permutedims(centered_digits, perm), (q.chunk_size * size(x, 1), size(x)[2:end]...))
end

"""
    sample_uniform_causal_chunk_mask(x, chunk_size)

Generate a mask of all the "chunks" towards the end of the sequence, separately for each batch.
The mask dims will be length-by-batch, but contiguous chunks of `chunk_size` will be always be masked together.
"""
function sample_uniform_causal_chunk_mask(x, chunk_size)
    s = size(x)
    chunks = s[2] ÷ chunk_size
    mask = cumsum(similar(x, 1, chunks, s[3]) .= 1, dims = 2) ./ (chunks + 1)
    threshes = similar(x, 1, 1, s[3])
    rand!(threshes)
    return repeat(mask .< threshes, inner = (1, chunk_size, 1))
end


#=
using Pkg
Pkg.activate(".")
using Revise
using Onion, Flux

target = randn(Float32, 20, 2);
q = FSQ(5, 2);

rx = randn(Float32, 20,2)
x = q(rx)
chunked = Onion.chunk(x, q)
unchunked = Onion.unchunk(chunked, q)
x == unchunked

params = similar(target) .= 0;
x = q(params);
sum(abs2, x .- target)

for i in 1:100
    l,gs = Flux.withgradient(params) do p
            x = q(p)
            sum(abs2, x .- target)
        end    
    println(l)
    params .= params .- 0.005 .* gs[1]
end
x = q(params)
=#


