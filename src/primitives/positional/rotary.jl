function rotary_pos_emb(::DefaultBackend, x::AbstractArray, cos::AbstractArray, sin::AbstractArray)
    d = size(x, 1)
    x1 = selectdim(x, 1, 1:d÷2)
    x2 = selectdim(x, 1, d÷2+1:d)
    return vcat(
        x1 .* cos .- x2 .* sin,
        x2 .* cos .+ x1 .* sin,
    )
end
