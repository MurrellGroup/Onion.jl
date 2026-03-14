function Onion._softmax(::NNopBackend,
    x::AbstractMatrix, ::Val{1}
)
    y = NNop.online_softmax(x)
    return y
end
