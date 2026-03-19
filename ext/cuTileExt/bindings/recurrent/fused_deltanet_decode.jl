function Onion.fused_deltanet_decode!(::cuTileBackend,
    O::AbstractArray{T,3},
    q_raw::AbstractArray{T,3}, k_raw::AbstractArray{T,3}, v::AbstractArray{T,3},
    alpha::AbstractMatrix{T}, beta_raw::AbstractMatrix{T}, z::AbstractArray{T,3},
    A_log::AbstractVector, dt_bias::AbstractVector, norm_weight::AbstractVector,
    state::AbstractArray{T,4};
    head_dim::Int, norm_eps=1f-6,
) where T
    fused_deltanet_decode_step!(O, q_raw, k_raw, v, alpha, beta_raw, z,
        A_log, dt_bias, norm_weight, state; norm_eps)
    return O, state
end

function Onion.fused_deltanet_decode(::cuTileBackend,
    q_raw::AbstractArray{T,3}, k_raw::AbstractArray{T,3}, v::AbstractArray{T,3},
    alpha::AbstractMatrix{T}, beta_raw::AbstractMatrix{T}, z::AbstractArray{T,3},
    A_log::AbstractVector, dt_bias::AbstractVector, norm_weight::AbstractVector,
    state::AbstractArray{T,4};
    head_dim::Int, norm_eps=1f-6,
) where T
    O = similar(v)
    fused_deltanet_decode_step!(O, q_raw, k_raw, v, alpha, beta_raw, z,
        A_log, dt_bias, norm_weight, state; norm_eps)
    return O, state
end
