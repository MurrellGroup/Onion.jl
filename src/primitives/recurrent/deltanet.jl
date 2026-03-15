using Einops: einsum, @einops_str

# Interface: ensure all args have batch dimension before hitting primitive
# Batched: q (Dk, H, B), beta (H, B), state (Dk, Dv, H, B) → pass through
function deltanet_recurrent_decode(b::Backend,
    q::AbstractArray{T,3}, k::AbstractArray{T,3}, v::AbstractArray{T,3},
    beta::AbstractArray{T,2}, gate::AbstractArray{T,2},
    state::AbstractArray{T,4},
) where T
    _deltanet_recurrent_decode(b, q, k, v, beta, gate, state)
end

# q/k/v have batch dim but beta/gate don't — unsqueeze beta/gate
function deltanet_recurrent_decode(b::Backend,
    q::AbstractArray{T,2}, k::AbstractArray{T,2}, v::AbstractArray{T,2},
    beta::AbstractArray{T,1}, gate::AbstractArray{T,1},
    state::AbstractArray{T,4},
) where T
    q, k, v, beta, gate = reshape.((q, k, v, beta, gate), einops"... -> ... 1")
    output, state = _deltanet_recurrent_decode(b, q, k, v, beta, gate, state)
    output = reshape(output, einops"... 1 -> ...")
    return output, state
end

# Primitive implementation
function _deltanet_recurrent_decode(::DefaultBackend,
    q::AbstractArray{T},    # (Dk, Hk, B)
    k::AbstractArray{T},    # (Dk, Hk, B)
    v::AbstractArray{T},    # (Dv, Hv, B)
    beta::AbstractArray{T}, # (Hv, B)
    gate::AbstractArray{T}, # (Hv, B)
    state::AbstractArray{T} # (Dk, Dv, H, B) — mutated in-place
) where T
    state .*= exp.(reshape(gate, einops"... -> 1 1 ..."))

    v_per_k = cld(size(v, 2), size(k, 2))
    if v_per_k > 1
        q, k = repeat.((q, k), einops"Dk Hk ... -> Dk (Hk r) ..."; r=v_per_k)
    end

    sᵀk = einsum(state, k, einops"Dk Dv h ..., Dk h ... -> 1 Dv h ...")

    k = reshape(k, einops"Dk ... -> Dk 1 ...")
    v = reshape(v, einops"Dv ... -> 1 Dv ...")
    beta = reshape(beta, einops"... -> 1 1 ...")
    state .+= k .* (beta .* (v .- sᵀk))

    # Output: o = S^T @ q
    output = einsum(state, q, einops"Dk Dv h ..., Dk h ... -> Dv h ...")

    return output, state
end
