function Onion._layer_norm(::NNkernelsBackend,
    x::AbstractMatrix, w::AbstractVector, b::AbstractVector, ::Val{1};
    eps
)
    y = NNkernels.layer_norm(x, w, b; ϵ=eps)
    return y
end
