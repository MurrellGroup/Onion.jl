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
    result = rotary_pos_emb(DefaultBackend(), x, cos, sin)
    copyto!(out, result)
    return out
end

function rotary_pos_emb(::DefaultBackend, x::AbstractArray, cos::AbstractArray, sin::AbstractArray)
    d = size(x, 1)
    x1 = selectdim(x, 1, 1:d÷2)
    x2 = selectdim(x, 1, d÷2+1:d)
    return vcat(
        x1 .* cos .- x2 .* sin,
        x2 .* cos .+ x1 .* sin,
    )
end
