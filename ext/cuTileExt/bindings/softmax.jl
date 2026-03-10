import ChainRulesCore as CRC

function Onion.softmax(::cuTileBackend,
    x::AbstractMatrix
)
    y = online_softmax(x)
    return y
end

function CRC.rrule(::typeof(Onion.softmax), ::cuTileBackend,
    x::AbstractMatrix
)
    y = online_softmax(x)
    function softmax_pullback(ȳ)
        x̄ = ∇online_softmax(unthunk(ȳ), y)
        return NoTangent(), NoTangent(), x̄
    end
    return y, softmax_pullback
end
