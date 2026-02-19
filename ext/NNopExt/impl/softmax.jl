using Rewrap: Keep, Split, (..)

function Onion.softmax(::NNopBackend,
    x::AbstractMatrix
)
    y = NNop.online_softmax(x)
    return y
end
