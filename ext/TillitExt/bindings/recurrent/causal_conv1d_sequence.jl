function Onion.causal_conv1d_sequence!(::TillitBackend,
    y::AbstractArray{<:Any,3},
    x::AbstractArray{<:Any,3},
    weight::AbstractMatrix,
    bias::Optional{AbstractVector};
    silu::Bool = true,
)
    function verify()
        tol = eltype(x) === Float32 ? 1e-3 : 1e-1
        y_ref = Onion.causal_conv1d_sequence(DefaultBackend(),
            Array(x), Array(weight), isnothing(bias) ? nothing : Array(bias); silu)
        function iscorrect()
            isapprox(Array(y), y_ref; atol=tol, rtol=tol)
        end
    end

    Tillit.causal_conv1d_sequence!(y, x, weight, bias; silu, verify)
    return y
end
