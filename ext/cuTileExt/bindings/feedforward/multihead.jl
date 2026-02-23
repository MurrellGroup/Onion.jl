# XXX: verify correctness
function multihead_ffn(Q, K, U, V; kws...)
    O = similar(Q)
    multihead_ffn!(O, Q, K, U, V; kws...)
    return O
end

# XXX: verify correctness
function ∇multihead_ffn(Ō, Q, K, U, V; kws...)
    Q̄, K̄, Ū, V̄ = similar(Q), similar(K), similar(U), similar(V)
    ∇multihead_ffn!(Q̄, K̄, Ū, V̄, Ō, Q, K, U, V; kws...)
end

function Onion.multihead_ffn(::cuTileBackend,
    Q, K, U, V, ::typeof(Onion.swish); kws...
)
    return multihead_ffn(Q, K, U, V)
end

function rrule(
    ::typeof(Onion.multihead_ffn), ::cuTileBackend,
    Q::AbstractMatrix,
    K::AbstractMatrix, U::AbstractMatrix, V::AbstractMatrix,
    ::typeof(Onion.swish);
    kws...
)
    O = multihead_ffn(Q, K, U, V; kws...)
    function glu_ffn_pullback(Ō)
        Q̄, K̄, Ū, V̄ = ∇multihead_ffn(unthunk(Ō), Q, K, U, V; kws...)
        return NoTangent(), NoTangent(), Q̄, K̄, Ū, V̄, NoTangent()
    end
    return O, glu_ffn_pullback
end
