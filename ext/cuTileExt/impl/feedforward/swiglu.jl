function Onion.glu_ffn(::cuTileBackend,
    Q, K, U, V, ::typeof(Onion.swish)
)
    return swiglu_ffn(Q, K, U, V)
end
