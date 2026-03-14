using NNlib: swish

function swiglu_ffn(X, Wᵍ, Wᵘ, Wᵈ; kws...)
    O = similar(X, size(Wᵈ, 1), size(X, 2))
    verify = () -> let
        O_ref = Onion.glu_ffn(DefaultBackend(), X, Wᵍ, Wᵘ, Wᵈ, swish)
        () -> isapprox(O, O_ref, rtol=1e-1)
    end
    cache = swiglu_ffn!(O, X, Wᵍ, Wᵘ, Wᵈ; verify, kws...)
    return O, cache
end

function ∇swiglu_ffn(Ō, X, Wᵍ, Wᵘ, Wᵈ, cache)
    X̄, W̄ᵍ, W̄ᵘ, W̄ᵈ = similar.((X, Wᵍ, Wᵘ, Wᵈ))
    verify_x = () -> let
        _, pb = Zygote.pullback(X) do X
            Onion.glu_ffn(DefaultBackend(), X, Wᵍ, Wᵘ, Wᵈ, swish)
        end
        X̄_ref, = pb(Ō)
        () -> isapprox(X̄, X̄_ref, rtol=1e-1)
    end
    verify_w_down = () -> let
        _, pb = Zygote.pullback(Wᵈ) do Wᵈ
            Onion.glu_ffn(DefaultBackend(), X, Wᵍ, Wᵘ, Wᵈ, swish)
        end
        W̄ᵈ_ref, = pb(Ō)
        () -> isapprox(W̄ᵈ, W̄ᵈ_ref, rtol=1e-1)
    end
    verify_w = () -> let
        _, pb = Zygote.pullback(Wᵍ, Wᵘ) do Wᵍ, Wᵘ
            Onion.glu_ffn(DefaultBackend(), X, Wᵍ, Wᵘ, Wᵈ, swish)
        end
        W̄ᵍ_ref, W̄ᵘ_ref = pb(Ō)
        () -> isapprox(W̄ᵍ, W̄ᵍ_ref, rtol=1e-1) &&
              isapprox(W̄ᵘ, W̄ᵘ_ref, rtol=1e-1)
    end
    ∇swiglu_ffn!(X̄, W̄ᵍ, W̄ᵘ, W̄ᵈ, Ō, X, Wᵍ, Wᵘ, Wᵈ, cache...;
        verify_x, verify_w_down, verify_w)
    return X̄, W̄ᵍ, W̄ᵘ, W̄ᵈ
end

function Onion._glu_ffn(::cuTileBackend,
    X::AbstractMatrix,
    Wᵍ::AbstractMatrix, Wᵘ::AbstractMatrix, Wᵈ::AbstractMatrix,
    ::typeof(swish);
    kws...
)
    O, = swiglu_ffn(X, Wᵍ, Wᵘ, Wᵈ; kws...)
    return O
end

function CRC.rrule(
    ::typeof(Onion._glu_ffn), ::cuTileBackend,
    X::AbstractMatrix,
    Wᵍ::AbstractMatrix, Wᵘ::AbstractMatrix, Wᵈ::AbstractMatrix,
    ::typeof(swish)
)
    O, cache = swiglu_ffn(X, Wᵍ, Wᵘ, Wᵈ)
    function glu_ffn_pullback(Ō)
        X̄, W̄... = ∇swiglu_ffn(unthunk(Ō), X, Wᵍ, Wᵘ, Wᵈ, cache)
        return NoTangent(), NoTangent(), X̄, W̄..., NoTangent()
    end
    return O, glu_ffn_pullback
end
