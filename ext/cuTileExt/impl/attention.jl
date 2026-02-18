@impl cuTileBackend function Primitives.attention(
    Q::AbstractArray,
    K::AbstractArray,
    V::AbstractArray;
    causal::Bool
)
    O, _, _ = flash_attention(Q, K, V; causal)
    return O
end
