function Onion._attention(::NNkernelsBackend,
    Q::AbstractArray, K::AbstractArray, V::AbstractArray;
    causal::Bool
)
    O = NNkernels.flash_attention(Q, K, V; causal)
    return O
end
