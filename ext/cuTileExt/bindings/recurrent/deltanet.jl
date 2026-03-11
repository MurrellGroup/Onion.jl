function Onion.deltanet_recurrent(::cuTileBackend,
    q::AbstractArray, k::AbstractArray, v::AbstractArray,
    beta::AbstractArray, gate::AbstractArray,
    state::AbstractArray,
)
    Dv = size(v, 1)
    O = similar(v)

    verify = () -> let
        state_copy = copy(state)
        O_ref, _ = Onion.deltanet_recurrent(DefaultBackend(),
            q → Float32, k → Float32, v → Float32,
            beta → Float32, gate → Float32, copy(state) → Float32)
        () -> isapprox(O, O_ref, rtol=1e-1)
    end

    deltanet_recurrent_step!(O, q, k, v, beta, gate, state; verify)
    return O, state
end
