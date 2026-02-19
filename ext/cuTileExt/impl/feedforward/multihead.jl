function Onion.multihead_ffn(::cuTileBackend,
    Q, K, U, V, ::typeof(Onion.swish)
)
    return multihead_ffn(Q, K, U, V)
end
