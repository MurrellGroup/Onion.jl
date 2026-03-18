using NNlib: sigmoid

"""
    causal_conv1d_sequence(x, weight, bias; silu) -> y

Full-sequence causal depthwise conv1d. Applies conv1d causally across the
time dimension (zero-padded left). No state — processes the whole sequence.

    x:      (D, T, B)
    weight: (D, K)
    y:      (D, T, B)
"""
@primitive _causal_conv1d_sequence as causal_conv1d_sequence
@primitive _causal_conv1d_sequence! as causal_conv1d_sequence!

function _causal_conv1d_sequence!(::DefaultBackend,
    y::AbstractArray{T,3},
    x::AbstractArray{T,3}, weight::AbstractArray{T}, bias::Optional{AbstractVector{T}};
    silu::Bool = true,
) where T
    result = _causal_conv1d_sequence(DefaultBackend(), x, weight, bias; silu)
    copyto!(y, result)
    return y
end

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
