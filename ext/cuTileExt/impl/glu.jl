@impl cuTileBackend function Onion.glu_ffn(
    Q, K, U, V, ::typeof(Onion.swish)
)
    return swiglu_ffn(Q, K, U, V)
end
