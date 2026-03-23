using NNlib: swish

function Onion.glu_ffn!(::TillitBackend,
    O, X, Wᵍ, Wᵘ, Wᵈ,
    ::typeof(Onion.swish);
    Gate = similar(X, K, N),
    Up   = similar(X, K, N),
    kws...
)
    function verify()
        O_ref = Onion.glu_ffn(DefaultBackend(), X, Wᵍ, Wᵘ, Wᵈ, swish)
        function iscorrect()
            isapprox(O, O_ref, atol=1e-1, rtol=1e-1)
        end
    end
    O = swiglu_ffn!(O, X, Wᵍ, Wᵘ, Wᵈ; Gate, Up, verify, kws...)
    return O
end

function CRC.rrule(::typeof(glu_ffn!), backend::TillitBackend,
    O, X, Wᵍ, Wᵘ, Wᵈ, ::typeof(Onion.swish);
    Gate = similar(X, K, N),
    Up   = similar(X, K, N),
    kws...
)
    Onion.glu_ffn!(backend, O, X, Wᵍ, Wᵘ, Wᵈ, Onion.swish; Gate, Up, kws...)
    function glu_ffn_pullback(Ō)
        Ō = unthunk(Ō)
        X̄, W̄ᵍ, W̄ᵘ, W̄ᵈ = similar.((X, Wᵍ, Wᵘ, Wᵈ))
        function verify_x()
            _, pb = Zygote.pullback(X) do X
                Onion.glu_ffn(DefaultBackend(), X, Wᵍ, Wᵘ, Wᵈ, swish)
            end
            X̄_ref, = pb(Ō)
            function iscorrect()
                isapprox(X̄, X̄_ref, atol=1e-1, rtol=1e-1)
            end
        end
        function verify_w_down()
            _, pb = Zygote.pullback(Wᵈ) do Wᵈ
                Onion.glu_ffn(DefaultBackend(), X, Wᵍ, Wᵘ, Wᵈ, swish)
            end
            W̄ᵈ_ref, = pb(Ō)
            function iscorrect()
                isapprox(W̄ᵈ, W̄ᵈ_ref, atol=1e-1, rtol=1e-1)
            end
        end
        function verify_w()
            _, pb = Zygote.pullback(Wᵍ, Wᵘ) do Wᵍ, Wᵘ
                Onion.glu_ffn(DefaultBackend(), X, Wᵍ, Wᵘ, Wᵈ, swish)
            end
            W̄ᵍ_ref, W̄ᵘ_ref = pb(Ō)
            function iscorrect()
                isapprox(W̄ᵍ, W̄ᵍ_ref, atol=1e-1, rtol=1e-1) &&
                isapprox(W̄ᵘ, W̄ᵘ_ref, atol=1e-1, rtol=1e-1)
            end
        end
        ∇swiglu_ffn!(X̄, W̄ᵍ, W̄ᵘ, W̄ᵈ, Ō, X, Wᵍ, Wᵘ, Wᵈ;
            Gate, Up,
            verify_x, verify_w_down, verify_w)
        return NoTangent(), NoTangent(), NoTangent(), X̄, W̄ᵍ, W̄ᵘ, W̄ᵈ
    end
    return O, glu_ffn_pullback
end
