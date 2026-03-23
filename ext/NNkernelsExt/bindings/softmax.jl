function Onion._softmax(::NNkernelsBackend,
    x::AbstractMatrix, ::Val{1}
)
    y = NNkernels.online_softmax(x)
    return y
end
