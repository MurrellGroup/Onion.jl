using Rewrap: Keep, Split, (..)

function Onion.layer_norm(::NNopBackend,
    x::AbstractMatrix, w::AbstractVector, b::AbstractVector;
    eps
)
    y = NNop.layer_norm(x, w, b; ϵ=eps)
    return y
end
