function linear(X, W, B; Y=similar(X, size(W, 1), size(X, 2)), kws...)
    function verify()
        Y_ref = Onion.linear(DefaultBackend(), X, W, B)
        function iscorrect()
            isapprox(Y, Y_ref, atol=1e-1, rtol=1e-1)
        end
    end
    linear!(W, B isa Bool ? nothing : B, X, Y; kws...)
    return Y
end

function ∇linear(Ȳ, X, W, B)
    X̄, W̄ = similar(X), similar(W)
    B̄ = B isa Bool ? nothing : similar(B)
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
    ∇linear!(X̄, W̄, B̄, Ȳ, W, X; verify_x, verify_w)
    return X̄, W̄, B̄
end

function Onion._linear(::cuTileBackend,
    X::AbstractMatrix, W::AbstractMatrix, B::Union{AbstractVector,Bool};
    kws...
)
    return linear(X, W, B; kws...)
end

function CRC.rrule(::typeof(Onion._linear), ::cuTileBackend,
    X::AbstractMatrix, W::AbstractMatrix, B::Union{AbstractVector,Bool},
)
    Y = linear(X, W, B)
    function linear_pullback(Ȳ)
        X̄, W̄, B̄ = ∇linear(unthunk(Ȳ), X, W, B)
        b_tangent = isnothing(B̄) ? NoTangent() : B̄
        return NoTangent(), NoTangent(), X̄, W̄, b_tangent
    end
    return Y, linear_pullback
end
