function Onion.fused_deltanet_decode!(::TillitBackend,
    O::AbstractArray{T,3},
    q_raw::AbstractArray{T,3}, k_raw::AbstractArray{T,3}, v::AbstractArray{T,3},
    alpha::AbstractMatrix{T}, beta_raw::AbstractMatrix{T}, z::AbstractArray{T,3},
    A_log::AbstractVector, dt_bias::AbstractVector, norm_weight::AbstractVector,
    state::AbstractArray{T,4};
    head_dim::Int, norm_eps=1f-6,
) where T
    function verify()
        tol = eltype(q_raw) === Float32 ? 1e-2 : 1e-1
        O_ref, state_ref = Onion.fused_deltanet_decode(DefaultBackend(),
            Array(q_raw), Array(k_raw), Array(v),
            Array(alpha), Array(beta_raw), Array(z),
            Array(A_log), Array(dt_bias), Array(norm_weight),
            Array(state); head_dim, norm_eps)
        function iscorrect()
            isapprox(Array(O), O_ref; atol=tol, rtol=tol) &&
            isapprox(Array(state), state_ref; atol=tol, rtol=tol)
        end
    end

    Tillit.fused_deltanet_decode!(O, q_raw, k_raw, v, alpha, beta_raw, z,
        A_log, dt_bias, norm_weight, state; norm_eps, verify)
    return O, state
end
