function Onion.attention(::NNopBackend,
    Q::AbstractArray, K::AbstractArray, V::AbstractArray;
    causal::Bool
)
    O = NNop.flash_attention(Q, K, V; causal)
    return O
end
