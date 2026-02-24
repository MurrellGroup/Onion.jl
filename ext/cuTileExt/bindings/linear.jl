function linear(X, W, B; Y=similar(X, size(W, 1), size(X, 2)), kws...)
    verify = () -> let
        Y_ref = Onion.linear(DefaultBackend(), X, W, B)
        () -> isapprox(Y, Y_ref, rtol=1e-1)
    end
    linear!(W, B isa Bool ? nothing : B, X, Y; kws...)
    return Y
end

function ∇linear(Ȳ, X, W, B)
    X̄, W̄ = similar(X), similar(W)
    B̄ = B isa Bool ? nothing : similar(B)
    verify_x = () -> let
        _, pb = Zygote.pullback(X→Float32, W→Float32, B→Float32) do X, W, B
            Onion.linear(DefaultBackend(), X, W, B)
        end
        X̄_ref, = pb(Ȳ)
        () -> isapprox(X̄, X̄_ref, rtol=1e-1)
    end
    verify_w = () -> let
        _, pb = Zygote.pullback(W, X, B) do W, X, B
            Onion.linear(DefaultBackend(), X, W, B)
        end
        W̄_ref, = pb(Ȳ)
        () -> isapprox(W̄, W̄_ref, rtol=1e-1)
    end
    ∇linear!(X̄, W̄, B̄, Ȳ, W, X; verify_x, verify_w)
    return X̄, W̄, B̄
end

function Onion.linear(::cuTileBackend,
    X::AbstractMatrix, W::AbstractMatrix, B::Union{AbstractVector,Bool};
    kws...
)
    return linear(X, W, B; kws...)
end

function CRC.rrule(::typeof(Onion.linear), ::cuTileBackend,
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
