using Rewrap: Keep, Split, (..)

@impl NNopBackend function Onion.rms_norm(
    x::AbstractArray, w::AbstractVector, b::AbstractVector;
    eps, dims = 1
)
    if dims === 1
        x′ = reshape(x, Keep(), :)
        y′ = NNop.layer_norm(x′, w, b; ϵ=eps)
        y = reshape(y′, Keep(), Split(.., size(x)[2:end]))
    else
        y = Onion.layer_norm(DefaultBackend(), x, w, b; eps, dims)
    end
    return y
end
