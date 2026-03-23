function Onion.deltanet_recurrent_decode!(::TillitBackend,
    O::AbstractArray{<:Any,3},
    q::AbstractArray{<:Any,3}, k::AbstractArray{<:Any,3}, v::AbstractArray{<:Any,3},
    beta::AbstractMatrix, gate::AbstractMatrix,
    state::AbstractArray{<:Any,4},
)
    function verify()
        tol = eltype(q) === Float32 ? 1e-3 : 1e-1
        O_ref, state_ref = Onion.deltanet_recurrent_decode(DefaultBackend(),
            Array(q), Array(k), Array(v), Array(beta), Array(gate), Array(state))
        function iscorrect()
            isapprox(Array(O), O_ref; atol=tol, rtol=tol) &&
            isapprox(Array(state), state_ref; atol=tol, rtol=tol)
        end
    end

    Tillit.deltanet_recurrent_decode!(O, q, k, v, beta, gate, state; verify)
    return O, state
end
