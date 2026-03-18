"""
    rotary_pos_emb(x, cos, sin)

Apply rotary positional embeddings. Splits `x` along dim 1 into halves and
applies the rotation: `[x₁·cos - x₂·sin; x₂·cos + x₁·sin]`.
"""
@primitive rotary_pos_emb
@primitive rotary_pos_emb!

function rotary_pos_emb!(::DefaultBackend,
    out::AbstractArray, x::AbstractArray, cos::AbstractArray, sin::AbstractArray
)
    d = size(x, 1)
    x1 = selectdim(x, 1, 1:d÷2)
    x2 = selectdim(x, 1, d÷2+1:d)
    out[1:d÷2, ntuple(_ -> Colon(), ndims(x)-1)...] .= x1 .* cos .- x2 .* sin
    out[d÷2+1:d, ntuple(_ -> Colon(), ndims(x)-1)...] .= x2 .* cos .+ x1 .* sin
    return out
end

get_buffers(::typeof(rotary_pos_emb), b::Backend, x, cos, sin) = (; out = similar(x))

function rotary_pos_emb(b::Backend, x::AbstractArray, cos::AbstractArray, sin::AbstractArray)
    bufs = get_buffers(rotary_pos_emb, b, x, cos, sin)
    rotary_pos_emb!(b, bufs.out, x, cos, sin)
    return bufs.out
end
