# Fused DeltaNet decode: L2norm(Q,K) + gate/beta + recurrence + RMSNorm + z-gate.
# Takes raw post-conv Q, K, V and all gate/norm parameters.
# Returns output ready for o_proj (already normed and z-gated).

function fused_deltanet_decode(b::Backend,
    q_raw::AbstractArray{T,3}, k_raw::AbstractArray{T,3}, v::AbstractArray{T,3},
    alpha::AbstractMatrix{T}, beta_raw::AbstractMatrix{T}, z::AbstractArray{T,3},
    A_log::AbstractVector, dt_bias::AbstractVector, norm_weight::AbstractVector,
    state::AbstractArray{T,4};
    head_dim::Int, norm_eps=T(1e-6),
) where T
    _fused_deltanet_decode(b, q_raw, k_raw, v, alpha, beta_raw, z,
        A_log, dt_bias, norm_weight, state; head_dim, norm_eps)
end

# Unbatched: (Dk, H) → add batch dim
function fused_deltanet_decode(b::Backend,
    q_raw::AbstractMatrix{T}, k_raw::AbstractMatrix{T}, v::AbstractMatrix{T},
    alpha::AbstractVector{T}, beta_raw::AbstractVector{T}, z::AbstractMatrix{T},
    A_log::AbstractVector, dt_bias::AbstractVector, norm_weight::AbstractVector,
    state::AbstractArray{T,4};
    head_dim::Int, norm_eps=T(1e-6),
) where T
    q3, k3, v3, z3 = reshape.((q_raw, k_raw, v, z), einops"... -> ... 1")
    alpha2, beta2 = reshape.((alpha, beta_raw), einops"... -> ... 1")
    output, state = _fused_deltanet_decode(b, q3, k3, v3, alpha2, beta2, z3,
        A_log, dt_bias, norm_weight, state; head_dim, norm_eps)
    return reshape(output, einops"... 1 -> ..."), state
end

using NNlib: sigmoid

function _fused_deltanet_decode(::DefaultBackend,
    q_raw::AbstractArray{T,3}, k_raw::AbstractArray{T,3}, v::AbstractArray{T,3},
    alpha::AbstractMatrix{T}, beta_raw::AbstractMatrix{T}, z::AbstractArray{T,3},
    A_log::AbstractVector, dt_bias::AbstractVector, norm_weight::AbstractVector,
    state::AbstractArray{T,4};
    head_dim::Int, norm_eps=T(1e-6),
) where T
    scale = T(inv(sqrt(Float64(head_dim))))
    q = q_raw ./ .√(sum(abs2, q_raw; dims=1) .+ T(1e-6)) .* scale
    k = k_raw ./ .√(sum(abs2, k_raw; dims=1) .+ T(1e-6))

    sp_input = alpha .+ dt_bias
    sp = @. ifelse(sp_input > T(20), sp_input, log1p(exp(sp_input)))
    gate = .-exp.(A_log) .* sp
    β = sigmoid.(beta_raw)

    output, state = _deltanet_recurrent_decode(DefaultBackend(), q, k, v, β, gate, state)

    # RMSNorm (not zero-centered) + z-gate
    rms = .√(sum(abs2, output; dims=1) ./ size(output, 1) .+ norm_eps)
    output = output ./ rms .* norm_weight
    zh = z .* sigmoid.(z)  # silu
    output = output .* zh

    return output, state
end
