using NNlib: sigmoid

# Interface: full sequence (D, T, B)
function causal_conv1d_sequence(b::Backend,
    x::AbstractArray{T,3},
    weight::AbstractArray{T,2},
    bias::Optional{AbstractVector{T}} = nothing;
    kws...
) where T
    _causal_conv1d_sequence(b, x, weight, bias; kws...)
end

# (D, T) unbatched → add batch dim
function causal_conv1d_sequence(b::Backend,
    x::AbstractArray{T,2},
    weight::AbstractArray{T,2},
    bias::Optional{AbstractVector{T}} = nothing;
    kws...
) where T
    y = _causal_conv1d_sequence(b, reshape(x, Keep(..), 1), weight, bias; kws...)
    return dropdims(y; dims=3)
end

function _causal_conv1d_sequence(::DefaultBackend,
    x::AbstractArray{T,3},    # (D, T, B)
    weight::AbstractArray{T}, # (D, K)
    bias::Optional{AbstractVector{T}};
    silu::Bool = true,
) where T
    D, L, B = size(x)
    K = size(weight, 2)
    y = similar(x)
    for t in 1:L
        t_start = max(1, t - K + 1)
        # weight[:, K] is newest, weight[:, 1] is oldest
        # At position t, the window is x[:, t_start:t, :]
        # with weight[:, K-t+t_start : K]
        acc = zeros(T, D, B)
        for (i, s) in enumerate(t_start:t)
            w_idx = K - (t - s)  # K for s=t, K-1 for s=t-1, etc.
            acc .+= x[:, s, :] .* @view(weight[:, w_idx:w_idx])
        end
        y[:, t, :] .= acc
    end
    isnothing(bias) || (y .+= reshape(bias, :, 1, 1))
    silu && (y .= y .* sigmoid.(y))
    return y
end
