@primitive glu_ffn
@primitive glu_ffn!

function glu_ffn!(::DefaultBackend,
    y::AbstractMatrix,
    x::AbstractMatrix,
    W_gate::AbstractMatrix, W_up::AbstractMatrix, W_down::AbstractMatrix,
    act = swish
)
    result = W_down * (act.(W_gate * x) .* (W_up * x))
    copyto!(y, result)
    return y
end

get_buffers(::typeof(glu_ffn), b::Backend, x, W_gate, W_up, W_down, act=nothing) =
    (; y = similar(x, size(W_down, 1), size(x, 2)))

function glu_ffn(b::Backend,
    x::AbstractMatrix,
    W_gate::AbstractMatrix, W_up::AbstractMatrix, W_down::AbstractMatrix,
    act = swish
)
    bufs = get_buffers(glu_ffn, b, x, W_gate, W_up, W_down, act)
    glu_ffn!(b, bufs.y, x, W_gate, W_up, W_down, act)
    return bufs.y
end
