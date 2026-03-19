function Onion.causal_conv1d!(::cuTileBackend,
    y::AbstractMatrix{T}, x::AbstractMatrix{T}, conv_state::AbstractArray{T,3},
    weight::AbstractMatrix{T}, bias::Optional{AbstractVector{T}};
    silu::Bool = true,
) where T
    causal_conv1d_step!(y, x, conv_state, weight, bias; silu)
    return y, conv_state
end

function Onion.causal_conv1d(::cuTileBackend,
    x::AbstractMatrix{T}, conv_state::AbstractArray{T,3},
    weight::AbstractMatrix{T}, bias::Optional{AbstractVector{T}};
    silu::Bool = true,
) where T
    Y = similar(x)
    causal_conv1d_step!(Y, x, conv_state, weight, bias; silu)
    return Y, conv_state
end
