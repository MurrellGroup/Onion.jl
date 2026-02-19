using Rewrap: Keep, Split, (..)

@impl NNopBackend function Onion.softmax(
    x::AbstractArray;
    dims = 1
)
    if dims === 1
        x′ = reshape(x, Keep(), :)
        y′ = NNop.online_softmax(x′)
        y = reshape(y′, Keep(), Split(.., size(x)[2:end]))
    else
        y = Onion.softmax(DefaultBackend(), x, dims)
    end
    return y
end
