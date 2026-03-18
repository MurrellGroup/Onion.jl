function Onion.fused_qknorm_rope(::cuTileBackend,
    q::AbstractArray{T,4}, k::AbstractArray{T,4},
    q_norm_weight, k_norm_weight,
    rope_cos, rope_sin;
    eps, offset, rotary_dim,
) where T
    Q = copy(q)  # will be mutated in-place
    K = copy(k)
    tol = T === Float32 ? 1e-3 : 1e-1

    # Cos/sin might be 2D (rotary_dim/2, 1) from rope[i:i] — flatten to 1D
    cos_1d = vec(rope_cos)
    sin_1d = vec(rope_sin)

    function verify()
        q_ref, k_ref = Onion.fused_qknorm_rope(DefaultBackend(),
            Array(q), Array(k), Array(q_norm_weight), Array(k_norm_weight),
            Array(cos_1d), Array(sin_1d); eps, offset, rotary_dim)
        function iscorrect()
            isapprox(Float32.(Array(Q)), Float32.(q_ref); atol=tol, rtol=tol) &&
            isapprox(Float32.(Array(K)), Float32.(k_ref); atol=tol, rtol=tol)
        end
    end

    fused_qknorm_rope_step!(Q, K, q_norm_weight, k_norm_weight, cos_1d, sin_1d;
        rotary_dim, eps=Float32(eps), offset=Float32(offset))
    return Q, K
end
