using Rewrap: Keep, Split, (..)

@impl NNopBackend function Onion.rms_norm(
    x::AbstractArray, w::AbstractVector;
    eps, offset = 0f0, dims = 1
)
    if dims === 1
        x′ = reshape(x, Keep(), :)
        y′ = NNop.rms_norm(x′, w; ϵ=eps, offset)
        y = reshape(y′, Keep(), Split(.., size(x)[2:end]))
    else
        y = Onion.rms_norm(DefaultBackend(), x, w)
    end
    return y
end
