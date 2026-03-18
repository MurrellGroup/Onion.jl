# Fused DeltaNet decode: L2norm(Q,K) + gate/beta + recurrence + RMSNorm + z-gate.
# Takes raw post-conv Q, K, V and all gate/norm parameters.
# Returns output ready for o_proj (already normed and z-gated).

"""
    fused_deltanet_decode(q_raw, k_raw, v, alpha, beta_raw, z,
                          A_log, dt_bias, norm_weight, state;
                          head_dim, norm_eps) -> (output, state)

Fused DeltaNet decode: L2-normalize Q/K, compute gate/beta, run recurrence,
apply RMSNorm + z-gate. All inputs must be batched (3D for q/k/v/z, 2D for scalars).
"""
@primitive fused_deltanet_decode
@primitive fused_deltanet_decode!

function fused_deltanet_decode!(::DefaultBackend,
    output::AbstractArray{T,3},
    q_raw::AbstractArray{T,3}, k_raw::AbstractArray{T,3}, v::AbstractArray{T,3},
    alpha::AbstractMatrix{T}, beta_raw::AbstractMatrix{T}, z::AbstractArray{T,3},
    A_log::AbstractVector, dt_bias::AbstractVector, norm_weight::AbstractVector,
    state::AbstractArray{T,4};
    head_dim::Int, norm_eps=T(1e-6),
) where T
    result, state = fused_deltanet_decode(DefaultBackend(),
        q_raw, k_raw, v, alpha, beta_raw, z,
        A_log, dt_bias, norm_weight, state; head_dim, norm_eps)
    copyto!(output, result)
    return output, state
end

get_buffers(::typeof(fused_deltanet_decode), b::Backend, q_raw, k_raw, v, args...; kws...) =
    (; output = similar(v))

using NNlib: sigmoid

function fused_deltanet_decode(::DefaultBackend,
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

    output, state = deltanet_recurrent_decode(DefaultBackend(), q, k, v, β, gate, state)

    # RMSNorm (not zero-centered) + z-gate
    rms = .√(sum(abs2, output; dims=1) ./ size(output, 1) .+ norm_eps)
    output = output ./ rms .* norm_weight
    zh = z .* sigmoid.(z)  # silu
    output = output .* zh

    return output, state
end
