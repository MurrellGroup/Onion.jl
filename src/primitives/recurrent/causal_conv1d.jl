using NNlib: sigmoid

"""
    causal_conv1d(x, conv_state, weight, bias; silu) -> (y, conv_state)

Causal depthwise conv1d state update for decode. Always batched:
x is `(D, B)`, conv_state is `(D, K, B)`.
"""
@primitive causal_conv1d
@primitive causal_conv1d!

function causal_conv1d!(::DefaultBackend,
    y::AbstractMatrix{T},
    x::AbstractMatrix{T}, conv_state::AbstractArray{T,3},
    weight::AbstractMatrix{T}, bias::Optional{AbstractVector{T}};
    silu::Bool = true,
) where T
    K = size(weight, 2)
    for i in 1:K-1
        conv_state[:, i, :] .= @view conv_state[:, i+1, :]
    end
    conv_state[:, K, :] .= x

    y .= einsum(conv_state, weight, einops"d k b, d k -> d b")
    isnothing(bias) || (y .+= bias)
    silu && (y .*= sigmoid.(y))

    return y, conv_state
end

get_buffers(::typeof(causal_conv1d), b::Backend, x, conv_state, weight, bias; kws...) =
    (; y = similar(x))

function causal_conv1d(b::Backend,
    x::AbstractMatrix{T}, conv_state::AbstractArray{T,3},
    weight::AbstractMatrix{T}, bias::Optional{AbstractVector{T}} = nothing;
    silu::Bool = true,
) where T
    bufs = get_buffers(causal_conv1d, b, x, conv_state, weight, bias; silu)
    causal_conv1d!(b, bufs.y, x, conv_state, weight, bias; silu)
    return bufs.y, conv_state
end
