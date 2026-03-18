@primitive _glu_ffn as glu_ffn
@primitive _glu_ffn! as glu_ffn!

function _glu_ffn!(::DefaultBackend,
    y::AbstractMatrix,
    x::AbstractMatrix,
    W_gate::AbstractMatrix, W_up::AbstractMatrix, W_down::AbstractMatrix,
    act = swish
)
    result = _glu_ffn(DefaultBackend(), x, W_gate, W_up, W_down, act)
    copyto!(y, result)
    return y
end

function _glu_ffn(::DefaultBackend,
    x::AbstractMatrix,
    W_gate::AbstractMatrix, W_up::AbstractMatrix, W_down::AbstractMatrix,
    act = swish
)
    y = W_down * (act.(W_gate * x) .* (W_up * x))
    return y
end
