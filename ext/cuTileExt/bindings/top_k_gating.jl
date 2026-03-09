function Onion.top_k_gating(::cuTileBackend, scores::AbstractMatrix, k::Integer)
    H, L = size(scores)

    I_out = similar(scores, Int32, k, L)
    V_out = similar(scores, k, L)

    top_k_gating_fwd!(I_out, V_out, scores; k)

    return I_out, V_out
end

function CRC.rrule(::typeof(Onion.top_k_gating), ::cuTileBackend,
    scores::AbstractMatrix, k::Integer,
)
    H, L = size(scores)

    I_out = similar(scores, Int32, k, L)
    V_out = similar(scores, k, L)

    top_k_gating_fwd!(I_out, V_out, scores; k)

    function top_k_gating_pullback(Δ)
        dV = unthunk(Δ[2])

        dS = fill!(similar(scores), 0)
        top_k_gating_bwd!(dS, dV, I_out; k)

        return NoTangent(), NoTangent(), dS, NoTangent()
    end

    return (I_out, V_out), top_k_gating_pullback
end
