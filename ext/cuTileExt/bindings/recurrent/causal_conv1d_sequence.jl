function Onion.causal_conv1d_sequence(::cuTileBackend,
    x::AbstractArray{<:Any,3},
    weight::AbstractMatrix,
    bias::Optional{AbstractVector};
    silu::Bool = true,
)
    Y = similar(x)

    tol = eltype(x) === Float32 ? 1e-3 : 1e-1

    function verify()
        Y_ref = Onion.causal_conv1d_sequence(DefaultBackend(),
            Array(x), Array(weight), isnothing(bias) ? nothing : Array(bias); silu)
        function iscorrect()
            isapprox(Array(Y), Y_ref; atol=tol, rtol=tol)
        end
    end

    causal_conv1d_sequence_step!(Y, x, weight, bias; silu, verify)
    return Y
end
