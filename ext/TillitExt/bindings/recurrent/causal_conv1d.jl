function Onion.causal_conv1d!(::TillitBackend,
    y::AbstractMatrix{T}, x::AbstractMatrix{T}, conv_state::AbstractArray{T,3},
    weight::AbstractMatrix{T}, bias::Optional{AbstractVector{T}};
    silu::Bool = true,
) where T
    function verify()
        tol = eltype(x) === Float32 ? 1e-3 : 1e-1
        y_ref, state_ref = Onion.causal_conv1d(DefaultBackend(),
            Array(x), Array(conv_state), Array(weight),
            isnothing(bias) ? nothing : Array(bias); silu)
        function iscorrect()
            isapprox(Array(y), y_ref; atol=tol, rtol=tol) &&
            isapprox(Array(conv_state), state_ref; atol=tol, rtol=tol)
        end
    end

    Tillit.causal_conv1d!(y, x, conv_state, weight, bias; silu, verify)
    return y, conv_state
end
