function Onion.fused_deltanet_decode(::cuTileBackend,
    q_raw::AbstractArray{<:Any,3}, k_raw::AbstractArray{<:Any,3}, v::AbstractArray{<:Any,3},
    alpha::AbstractMatrix, beta_raw::AbstractMatrix, z::AbstractArray{<:Any,3},
    A_log::AbstractVector, dt_bias::AbstractVector, norm_weight::AbstractVector,
    state::AbstractArray{<:Any,4};
    head_dim::Int, norm_eps=1f-6,
)
    O = similar(v)
    tol = eltype(q_raw) === Float32 ? 1e-3 : 1e-1

    function verify()
        O_ref, S_ref = Onion.fused_deltanet_decode(DefaultBackend(),
            Array(q_raw), Array(k_raw), Array(v),
            Array(alpha), Array(beta_raw), Array(z),
            Array(A_log), Array(dt_bias), Array(norm_weight),
            copy(Array(state)); head_dim, norm_eps)
        function iscorrect()
            isapprox(Float32.(Array(O)), Float32.(O_ref); atol=tol, rtol=tol) &&
            isapprox(Float32.(Array(state)), Float32.(S_ref); atol=tol, rtol=tol)
        end
    end

    fused_deltanet_decode_step!(O, q_raw, k_raw, v, alpha, beta_raw, z,
        A_log, dt_bias, norm_weight, state; norm_eps)
    return O, state
end
