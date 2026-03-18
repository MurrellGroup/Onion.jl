using NNlib: sigmoid

"""
    causal_conv1d_sequence(x, weight, bias; silu) -> y

Full-sequence causal depthwise conv1d. Always batched:
x is `(D, T, B)`, weight is `(D, K)`.
"""
@primitive _causal_conv1d_sequence as causal_conv1d_sequence
@primitive _causal_conv1d_sequence! as causal_conv1d_sequence!

function _causal_conv1d_sequence!(::DefaultBackend,
    y::AbstractArray{T,3},
    x::AbstractArray{T,3}, weight::AbstractMatrix{T}, bias::Optional{AbstractVector{T}};
    silu::Bool = true,
) where T
    result = _causal_conv1d_sequence(DefaultBackend(), x, weight, bias; silu)
    copyto!(y, result)
    return y
end

function _causal_conv1d_sequence(::DefaultBackend,
    x::AbstractArray{T,3},    # (D, T, B)
    weight::AbstractMatrix{T}, # (D, K)
    bias::Optional{AbstractVector{T}};
    silu::Bool = true,
) where T
    D, L, B = size(x)
    K = size(weight, 2)
    y = similar(x)
    for t in 1:L
        t_start = max(1, t - K + 1)
        acc = zeros(T, D, B)
        for (i, s) in enumerate(t_start:t)
            w_idx = K - (t - s)
            acc .+= x[:, s, :] .* @view(weight[:, w_idx:w_idx])
        end
        y[:, t, :] .= acc
    end
    isnothing(bias) || (y .+= reshape(bias, :, 1, 1))
    silu && (y .= y .* sigmoid.(y))
    return y
end
