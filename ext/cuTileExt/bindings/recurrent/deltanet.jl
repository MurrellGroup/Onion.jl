function Onion._deltanet_recurrent_decode(::cuTileBackend,
    q::AbstractArray{<:Any,3}, k::AbstractArray{<:Any,3}, v::AbstractArray{<:Any,3},
    beta::AbstractMatrix, gate::AbstractMatrix,
    state::AbstractArray{<:Any,4},
)
    O = similar(v)
    tol = eltype(q) === Float32 ? 1e-3 : 1e-1

    function verify()
        O_ref, state_ref = Onion._deltanet_recurrent_decode(DefaultBackend(),
            q, k, v, beta, gate, copy(state))
        function iscorrect()
            isapprox(O, O_ref; atol=tol, rtol=tol) &&
            isapprox(state, state_ref; atol=tol, rtol=tol)
        end
    end

    deltanet_recurrent_decode_step!(O, q, k, v, beta, gate, state; verify)
    return O, state
end
