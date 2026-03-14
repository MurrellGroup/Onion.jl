function _glu_ffn(::DefaultBackend,
    x::AbstractMatrix,
    W_gate::AbstractMatrix, W_up::AbstractMatrix, W_down::AbstractMatrix,
    act = swish
)
    y = W_down * (act.(W_gate * x) .* (W_up * x))
    return y
end
