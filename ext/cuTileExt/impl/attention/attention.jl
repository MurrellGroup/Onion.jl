function flash_attention(Q, K, V; kws...)
    O = similar(Q, size(V,1), size(Q)[2:end]...)
    M, L = flash_attention!(O,
        Q, K, V; kws...,
        verify = () -> let
            O_ref = Onion.attention(DefaultBackend(), Q, K, V; kws...)
            () -> isapprox(O, O_ref, rtol=1e-2)
        end
    )
    return O, M, L
end

using Flux.Zygote

function ∇flash_attention(Ō, Q, K, V, O, M, L; kws...)
    Q̄, K̄, V̄ = similar.((Q, K, V))
    verify = () -> let
        _, pb = Zygote.pullback(Q, K, V) do Q, K, V
            Onion.attention(DefaultBackend(), Q, K, V; kws...)
        end
        Q̄_ref, K̄_ref, V̄_ref = pb(Ō)
        () -> isapprox(Q̄, Q̄_ref, rtol=5e-2) &&
              isapprox(K̄, K̄_ref, rtol=5e-2) &&
              isapprox(V̄, V̄_ref, rtol=5e-2)
    end
    ∇flash_attention!(Q̄, K̄, V̄, Ō, Q, K, V, O, M, L; kws..., verify)
    return Q̄, K̄, V̄
end

function Onion.attention(::cuTileBackend,
    Q::AbstractArray, K::AbstractArray, V::AbstractArray;
    kws...
)
    O, = flash_attention(Q, K, V; kws...)
    return O
end

function CRC.rrule(
    ::typeof(Onion.attention), ::cuTileBackend,
    Q::AbstractArray, K::AbstractArray, V::AbstractArray;
    kws...
)
    O, M, L = flash_attention(Q, K, V; kws...)
    function attention_pullback(Ō)
        Q̄, K̄, V̄ = ∇flash_attention(CRC.unthunk(Ō), Q, K, V, O, M, L; kws...)
        return CRC.NoTangent(), CRC.NoTangent(), Q̄, K̄, V̄
    end
    return O, attention_pullback
end
