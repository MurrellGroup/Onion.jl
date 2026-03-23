function Onion.softmax!(::TillitBackend, y::AbstractMatrix, x::AbstractMatrix)
    Tillit.softmax!(y, x)
    return y
end

function CRC.rrule(::typeof(Onion.softmax!), backend::TillitBackend,
    y::AbstractMatrix,
    x::AbstractMatrix
)
    y = Onion.softmax!(backend, y, x)
    function softmax_pullback(ȳ)
        x̄ = similar(X)
        x̄ = Tillit.∇softmax!(x̄, unthunk(ȳ), y)
        return NoTangent(), NoTangent(), NoTangent(), x̄
    end
    return y, softmax_pullback
end
