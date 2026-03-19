function Onion.softmax!(::cuTileBackend, y::AbstractMatrix, x::AbstractMatrix)
    softmax!(y, x)
    return y
end

function CRC.rrule(::typeof(Onion.softmax!), ::cuTileBackend,
    y::AbstractMatrix,
    x::AbstractMatrix
)
    softmax!(y, x)
    function softmax_pullback(ȳ)
        x̄ = ∇softmax(unthunk(ȳ), y)
        return NoTangent(), NoTangent(), NoTangent(), x̄
    end
    return y, softmax_pullback
end
