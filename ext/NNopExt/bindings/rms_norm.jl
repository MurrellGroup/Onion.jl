function Onion._rms_norm(::NNopBackend,
    x::AbstractMatrix, w::AbstractVector, ::Val{1};
    eps, offset = 0f0
)
    y = NNop.rms_norm(x, w; ϵ=eps, offset)
    return y
end
