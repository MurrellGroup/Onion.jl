using Rewrap: Keep, Split, (..)

function Onion.rms_norm(::NNopBackend,
    x::AbstractMatrix, w::AbstractVector;
    eps, offset = 0f0
)
    y = NNop.rms_norm(x, w; ϵ=eps, offset)
    return y
end
