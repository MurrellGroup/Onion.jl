import ChainRulesCore as CRC

function Onion._softmax(::cuTileBackend,
    x::AbstractMatrix, ::Val{1}
)
    y = online_softmax(x)
    return y
end

function CRC.rrule(::typeof(Onion._softmax), ::cuTileBackend,
    x::AbstractMatrix, ::Val{1}
)
    y = online_softmax(x)
    function softmax_pullback(ȳ)
        x̄ = ∇online_softmax(unthunk(ȳ), y)
        return NoTangent(), NoTangent(), x̄, NoTangent()
    end
    return y, softmax_pullback
end
