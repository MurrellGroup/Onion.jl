function Onion.linear!(::TillitBackend,
    Y::AbstractMatrix,
    X::AbstractMatrix, W::AbstractMatrix, B::Optional{AbstractVector};
    kws...
)
    function verify()
        Y_ref = Onion.linear(DefaultBackend(), X, W, B)
        function iscorrect()
            isapprox(Y, Y_ref, atol=1e-1, rtol=1e-1)
        end
    end
    Tillit.linear!(W, B, X, Y; verify, kws...)
    return Y
end

function CRC.rrule(::typeof(Onion.linear!), backend::TillitBackend,
    Y::AbstractMatrix,
    X::AbstractMatrix, W::AbstractMatrix, B::Optional{AbstractVector}
)
    Y = Onion.linear!(backend, Y, X, W, B)
    function linear_pullback(Ȳ)
        X̄, W̄ = similar(X), similar(W)
        B̄ = isnothing(B) ? nothing : similar(B)
        function verify_x()
            _, pb = Zygote.pullback(X→Float32, W→Float32, B→Float32) do X, W, B
                Onion.linear(DefaultBackend(), X, W, B)
            end
            X̄_ref, = pb(Ȳ)
            function iscorrect()
                isapprox(X̄, X̄_ref, atol=1e-1, rtol=1e-1)
            end
        end
        function verify_w()
            _, pb = Zygote.pullback(W, X, B) do W, X, B
                Onion.linear(DefaultBackend(), X, W, B)
            end
            W̄_ref, = pb(Ȳ)
            function iscorrect()
                isapprox(W̄, W̄_ref, atol=1e-1, rtol=1e-1)
            end
        end
        Tillit.∇linear!(X̄, W̄, B̄, Ȳ, W, X; verify_x, verify_w)
        b_tangent = isnothing(B̄) ? NoTangent() : B̄
        return NoTangent(), NoTangent(), NoTangent(), X̄, W̄, b_tangent
    end
    return Y, linear_pullback
end
