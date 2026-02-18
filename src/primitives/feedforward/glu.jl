using ..Onion: swish

@impl DefaultBackend function glu_ffn(
    x::AbstractMatrix,
    W_up::AbstractMatrix, W_gate::AbstractMatrix, W_down::AbstractMatrix,
    act = swish
)
    y = W_down * (act.(W_gate * x) .* (W_up * x))
    return y
end
