@impl cuTileBackend function Primitives.multihead_ffn(
    Q, K, U, V, ::typeof(Onion.swish)
)
    return multihead_ffn(Q, K, U, V)
end
