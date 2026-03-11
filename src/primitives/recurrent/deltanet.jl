using Einops: einsum, @einops_str

function deltanet_recurrent(::DefaultBackend,
    q::AbstractArray{T},    # (Dk, H, B)
    k::AbstractArray{T},    # (Dk, H, B)
    v::AbstractArray{T},    # (Dv, H, B)
    beta::AbstractArray{T}, # (H, B)
    gate::AbstractArray{T}, # (H, B)
    state::AbstractArray{T} # (Dk, Dv, H, B) — mutated in-place
) where T
    # Decay: S *= exp(gate) broadcast over Dk, Dv dims
    decay = exp.(reshape(gate, 1, 1, size(gate)...))
    state .*= decay

    # S^T @ k: contract over Dk dimension
    stk = einsum(state, k, einops"dk dv h b, dk h b -> dv h b")

    # Delta rule: delta = beta * (v - S^T @ k)
    delta = reshape(beta, 1, size(beta)...) .* (v .- stk)

    # Rank-1 update: S += k ⊗ delta
    state .+= einsum(k, delta, einops"dk h b, dv h b -> dk dv h b")

    # Output: o = S^T @ q
    output = einsum(state, q, einops"dk dv h b, dk h b -> dv h b")

    return output, state
end
