function Onion.causal_conv1d(::cuTileBackend,
    x::AbstractArray, conv_state::AbstractArray,
    weight::AbstractArray, bias::Union{AbstractVector, Bool} = false;
    silu::Bool = true,
)
    Y = similar(x)
    bias_arr = bias === false ? nothing : bias

    verify = () -> let
        Y_ref, _ = Onion.causal_conv1d(DefaultBackend(),
            x → Float32, copy(conv_state) → Float32,
            weight → Float32, bias === false ? false : bias → Float32;
            silu)
        () -> isapprox(Y, Y_ref, rtol=1e-1)
    end

    causal_conv1d_step!(Y, x, conv_state, weight, bias_arr; silu, verify)
    return Y, conv_state
end
