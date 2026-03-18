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
    result, conv_state = causal_conv1d(DefaultBackend(), x, conv_state, weight, bias; silu)
    copyto!(y, result)
    return y, conv_state
end

function causal_conv1d(::DefaultBackend,
    x::AbstractMatrix{T},          # (D, B)
    conv_state::AbstractArray{T,3}, # (D, K, B) — mutated in-place
    weight::AbstractMatrix{T},      # (D, K)
    bias::Optional{AbstractVector{T}};
    silu::Bool = true,
) where T
    K = size(weight, 2)
    for i in 1:K-1
        conv_state[:, i, :] .= @view conv_state[:, i+1, :]
    end

    conv_state[:, K, :] .= x

    y = einsum(conv_state, weight, einops"d k b, d k -> d b")

    isnothing(bias) || (y .+= bias)
    silu && (y .*= sigmoid.(y))

    return y, conv_state
end
