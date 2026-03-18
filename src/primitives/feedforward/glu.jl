@primitive glu_ffn
@primitive glu_ffn!

function glu_ffn!(::DefaultBackend,
    y::AbstractMatrix,
    x::AbstractMatrix,
    W_gate::AbstractMatrix, W_up::AbstractMatrix, W_down::AbstractMatrix,
    act = swish
)
    result = glu_ffn(DefaultBackend(), x, W_gate, W_up, W_down, act)
    copyto!(y, result)
    return y
end

get_buffers(::typeof(glu_ffn), b::Backend, x, W_gate, W_up, W_down, act=nothing) =
    (; y = similar(x, size(W_down, 1), size(x, 2)))

function glu_ffn(::DefaultBackend,
    x::AbstractMatrix,
    W_gate::AbstractMatrix, W_up::AbstractMatrix, W_down::AbstractMatrix,
    act = swish
)
    y = W_down * (act.(W_gate * x) .* (W_up * x))
    return y
end
