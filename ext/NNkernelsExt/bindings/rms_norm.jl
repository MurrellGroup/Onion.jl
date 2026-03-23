function Onion._rms_norm(::NNkernelsBackend,
    x::AbstractMatrix, w::AbstractVector, ::Val{1};
    eps, offset = 0f0
)
    y = NNkernels.rms_norm(x, w; ϵ=eps, offset)
    return y
end
