function Onion._causal_conv1d(::cuTileBackend,
    x::AbstractMatrix, conv_state::AbstractArray{<:Any,3},
    weight::AbstractMatrix, bias::Optional{AbstractVector};
    silu::Bool = true,
)
    Y = similar(x)

    function verify()
        Y_ref, state_ref = Onion._causal_conv1d(DefaultBackend(),
            x, copy(conv_state), weight, bias; silu)
        function iscorrect()
            isapprox(Y, Y_ref, atol=1e-3, rtol=1e-3) &&
            isapprox(conv_state, state_ref, atol=1e-3, rtol=1e-3)
        end
    end

    causal_conv1d_step!(Y, x, conv_state, weight, bias; silu, verify)
    return Y, conv_state
end
