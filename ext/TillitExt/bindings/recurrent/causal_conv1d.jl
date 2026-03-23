function Onion.causal_conv1d!(::TillitBackend,
    y::AbstractMatrix{T}, x::AbstractMatrix{T}, conv_state::AbstractArray{T,3},
    weight::AbstractMatrix{T}, bias::Optional{AbstractVector{T}};
    silu::Bool = true,
) where T
    Tillit.causal_conv1d!(y, x, conv_state, weight, bias; silu)
    return y, conv_state
end
