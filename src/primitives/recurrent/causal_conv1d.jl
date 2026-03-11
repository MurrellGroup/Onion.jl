using NNlib: sigmoid

function causal_conv1d(::DefaultBackend,
    x::AbstractArray{T},          # (D, B) — new input
    conv_state::AbstractArray{T}, # (D, K, B) — ring buffer, mutated in-place
    weight::AbstractArray{T},     # (D, K) — conv weights
    bias::Union{AbstractVector{T}, Bool} = false;
    silu::Bool = true,
) where T
    K = size(conv_state, 2)

    # Shift left: state[:, i, :] = state[:, i+1, :] for i in 1:K-1
    for i in 1:K-1
        conv_state[:, i, :] .= @view conv_state[:, i+1, :]
    end
    # Insert new input at the end
    conv_state[:, K, :] .= x

    # Convolve: y = sum(state .* weight, dims=2)
    y = sum(conv_state .* reshape(weight, size(weight, 1), K, 1), dims=2)
    y = dropdims(y, dims=2)

    # Bias
    if bias !== false
        y = y .+ bias
    end

    # SiLU activation: x * sigmoid(x)
    if silu
        y = y .* sigmoid.(y)
    end

    return y, conv_state
end
