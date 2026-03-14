function _rotary_pos_emb(::DefaultBackend, x::AbstractArray, cos::AbstractArray, sin::AbstractArray)
    d = size(x, 1)
    x1 = selectdim(x, 1, 1:d÷2)
    x2 = selectdim(x, 1, d÷2+1:d)
    return vcat(
        x1 .* cos .- x2 .* sin,
        x2 .* cos .+ x1 .* sin,
    )
end

#=
function CRC.rrule(::typeof(rotary_pos_emb_forward),
                   x::CuArray, cos::AbstractArray, sin::AbstractArray)
    y = apply_rotary_pos_emb_fused(x, cos, sin)
    function rotary_pullback(dy_raw)
        dy = unthunk(dy_raw)
        # Inverse rotation: negate sin
        dx = apply_rotary_pos_emb_fused(dy, cos, .-sin)
        return NoTangent(), dx, NoTangent(), NoTangent()
    end
    return y, rotary_pullback
end
=#