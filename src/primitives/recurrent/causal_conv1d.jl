using NNlib: sigmoid

# Interface: x (D, B) with conv_state (D, K, B) → pass through
function causal_conv1d(b::Backend,
    x::AbstractArray{T,2}, conv_state::AbstractArray{T,3},
    weight::AbstractArray{T}, bias::Optional{AbstractVector{T}} = nothing;
    kws...
) where T
    _causal_conv1d(b, x, conv_state, weight, bias; kws...)
end

# x (D,) unbatched but conv_state (D, K, B) → unsqueeze x
function causal_conv1d(b::Backend,
    x::AbstractArray{T,1}, conv_state::AbstractArray{T,3},
    weight::AbstractArray{T}, bias::Optional{AbstractVector{T}} = nothing;
    kws...
) where T
    x = reshape(x, :, 1)
    y, conv_state = _causal_conv1d(b, x, conv_state, weight, bias; kws...)
    return dropdims(y; dims=2), conv_state
end

function _causal_conv1d(::DefaultBackend,
    x::AbstractArray{T},          # (D, B) — new input
    conv_state::AbstractArray{T}, # (D, K, B) — ring buffer, mutated in-place
    weight::AbstractArray{T},     # (D, K) — conv weights
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
