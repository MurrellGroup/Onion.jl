function Onion.deltanet_sequence!(::TillitBackend,
    O::AbstractArray{<:Any,4}, S::AbstractArray{<:Any,4},
    q::AbstractArray{<:Any,4}, k::AbstractArray{<:Any,4}, v::AbstractArray{<:Any,4},
    beta::AbstractArray{<:Any,3}, gate::AbstractArray{<:Any,3},
    initial_state::Optional{AbstractArray{<:Any,4}} = S,
)
    if !isnothing(initial_state) && initial_state !== S
        copy!(S, initial_state)
    elseif isnothing(initial_state)
        S .= 0
    end

    function verify()
        tol = eltype(q) === Float32 ? 1e-2 : 1e-1
        O_ref, S_ref = Onion.deltanet_sequence(DefaultBackend(),
            Array(q), Array(k), Array(v), Array(beta), Array(gate), Array(S))
        function iscorrect()
            isapprox(Array(O), O_ref; atol=tol, rtol=tol) &&
            isapprox(Array(S), S_ref; atol=tol, rtol=tol)
        end
    end

    Tillit.deltanet_sequence!(O, q, k, v, beta, gate, S; verify)
    return O, S
end
