using NNlib: swish

function multihead_ffn(Q, K, U, V; R = nothing, kws...)
    O = similar(Q)
    multihead_ffn!(O, Q, K, U, V; R, kws...)
    return O
end

function ∇multihead_ffn(Ō, Q, K, U, V; R = nothing, kws...)
    Q̄, K̄, Ū, V̄ = similar(Q), similar(K), similar(U), similar(V)
    R̄ = isnothing(R) ? nothing : similar(R)
    ∇multihead_ffn!(Q̄, K̄, Ū, V̄, Ō, Q, K, U, V; R, R̄, D_E, kws...)
    return isnothing(R) ? (Q̄, K̄, Ū, V̄) : (Q̄, K̄, Ū, V̄, R̄)
end

# ── Non-expert dispatch ──────────────────────────────────────────────

function Onion.multihead_ffn(::cuTileBackend,
    Q, K, U, V, ::typeof(swish)
)
    return multihead_ffn(Q, K, U, V)
end

function CRC.rrule(
    ::typeof(Onion.multihead_ffn), ::cuTileBackend,
    Q::AbstractArray, K::AbstractArray, U::AbstractArray, V::AbstractArray,
    ::typeof(swish)
)
    O = multihead_ffn(Q, K, U, V)
    function mhffn_pullback(Ō)
        Q̄, K̄, Ū, V̄ = ∇multihead_ffn(unthunk(Ō), Q, K, U, V)
        return NoTangent(), NoTangent(), Q̄, K̄, Ū, V̄, NoTangent()
    end
    return O, mhffn_pullback
end

# ── Expert dispatch ──────────────────────────────────────────────────

function Onion.multihead_ffn(::cuTileBackend,
    Q, K, U, V, ::typeof(swish), R
)
    return multihead_ffn(Q, K, U, V; R)
end

function CRC.rrule(
    ::typeof(Onion.multihead_ffn), ::cuTileBackend,
    Q::AbstractArray, K::AbstractArray, U::AbstractArray, V::AbstractArray,
    ::typeof(swish), R::AbstractArray
)
    O = multihead_ffn(Q, K, U, V; R)
    function mhffn_expert_pullback(Ō)
        Q̄, K̄, Ū, V̄, R̄ = ∇multihead_ffn(unthunk(Ō), Q, K, U, V; R)
        return NoTangent(), NoTangent(), Q̄, K̄, Ū, V̄, NoTangent(), R̄
    end
    return O, mhffn_expert_pullback
end
