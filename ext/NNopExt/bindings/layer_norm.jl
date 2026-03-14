function Onion._layer_norm(::NNopBackend,
    x::AbstractMatrix, w::AbstractVector, b::AbstractVector, ::Val{1};
    eps
)
    y = NNop.layer_norm(x, w, b; ϵ=eps)
    return y
end
