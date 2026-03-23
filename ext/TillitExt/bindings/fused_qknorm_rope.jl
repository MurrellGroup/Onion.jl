function Onion.fused_qknorm_rope!(::TillitBackend,
    q_out::AbstractArray{T,4}, k_out::AbstractArray{T,4},
    q::AbstractArray{T,4}, k::AbstractArray{T,4},
    q_norm_weight::AbstractVector, k_norm_weight::AbstractVector,
    rope_cos::AbstractArray, rope_sin::AbstractArray;
    eps, offset, rotary_dim,
) where T
    copyto!(q_out, q)
    copyto!(k_out, k)
    cos_1d = vec(rope_cos)
    sin_1d = vec(rope_sin)

    # XXX: propagate?
    function verify()
        q_ref, k_ref = Onion.fused_qknorm_rope(DefaultBackend(),
            Array(q), Array(k), Array(q_norm_weight), Array(k_norm_weight),
            Array(cos_1d), Array(sin_1d); eps, offset, rotary_dim)
        function iscorrect()
            isapprox(Float32.(Array(Q)), Float32.(q_ref); atol=tol, rtol=tol) &&
            isapprox(Float32.(Array(K)), Float32.(k_ref); atol=tol, rtol=tol)
        end
    end

    Tillitfused_qknorm_rope!(q_out, k_out, q_norm_weight, k_norm_weight, cos_1d, sin_1d;
        rotary_dim, eps=Float32(eps), offset=Float32(offset))
    return q_out, k_out
end
