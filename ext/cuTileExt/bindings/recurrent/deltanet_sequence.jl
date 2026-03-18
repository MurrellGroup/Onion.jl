function Onion._deltanet_sequence(::cuTileBackend,
    q::AbstractArray{<:Any,4}, k::AbstractArray{<:Any,4}, v::AbstractArray{<:Any,4},
    beta::AbstractArray{<:Any,3}, gate::AbstractArray{<:Any,3},
    initial_state::Optional{AbstractArray{<:Any,4}},
)
    Dk, T, H, B = size(q)
    Dv = size(v, 1)

    S = if isnothing(initial_state)
        fill!(similar(q, eltype(q), Dk, Dv, H, B), 0)
    else
        copy(initial_state)
    end

    O = similar(v)

    tol = eltype(q) === Float32 ? 1e-2 : 1e-1

    function verify()
        O_ref, S_ref = Onion._deltanet_sequence(DefaultBackend(),
            Array(q), Array(k), Array(v), Array(beta), Array(gate),
            isnothing(initial_state) ? nothing : Array(initial_state))
        function iscorrect()
            isapprox(Array(O), O_ref; atol=tol, rtol=tol) &&
            isapprox(Array(S), S_ref; atol=tol, rtol=tol)
        end
    end

    deltanet_sequence_step!(O, q, k, v, beta, gate, S; verify)
    return O, S
end
