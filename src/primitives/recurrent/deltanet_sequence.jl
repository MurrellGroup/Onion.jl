using Einops: einsum, @einops_str

# Interface: batched — q (Dk, T, H, B)
function deltanet_sequence(b::Backend,
    q::AbstractArray{T,4}, k::AbstractArray{T,4}, v::AbstractArray{T,4},
    beta::AbstractArray{T,3}, gate::AbstractArray{T,3},
    initial_state::Optional{AbstractArray{T,4}} = nothing,
) where T
    _deltanet_sequence(b, q, k, v, beta, gate, initial_state)
end

# Unbatched: q (Dk, T, H) → add batch dim
function deltanet_sequence(b::Backend,
    q::AbstractArray{T,3}, k::AbstractArray{T,3}, v::AbstractArray{T,3},
    beta::AbstractArray{T,2}, gate::AbstractArray{T,2},
    initial_state::Optional{AbstractArray{T,4}} = nothing,
) where T
    q4, k4, v4 = reshape.((q, k, v), einops"... -> ... 1")
    beta3, gate3 = reshape.((beta, gate), einops"... -> ... 1")
    o, final_state = _deltanet_sequence(b, q4, k4, v4, beta3, gate3, initial_state)
    return reshape(o, einops"... 1 -> ..."), final_state
end

# Naive sequential recurrence. Correct, simple, AD-friendly (no mutation).
# For GPU performance, a chunkwise WY backend would override this.
function _deltanet_sequence(::DefaultBackend,
    q::AbstractArray{T,4},    # (Dk, L, Hk, B)
    k::AbstractArray{T,4},    # (Dk, L, Hk, B)
    v::AbstractArray{T,4},    # (Dv, L, Hv, B)
    beta::AbstractArray{T,3}, # (Hv, L, B)
    gate::AbstractArray{T,3}, # (Hv, L, B)
    initial_state::Optional{AbstractArray{T,4}}, # (Dk, Dv, Hv, B) or nothing
) where T
    Dk, L, Hk, B = size(q)
    Dv, _, Hv, _ = size(v)

    # Handle multi-value-head per key-head
    v_per_k = cld(Hv, Hk)
    if v_per_k > 1
        q = repeat(q, einops"Dk L Hk B -> Dk L (Hk r) B"; r=v_per_k)
        k = repeat(k, einops"Dk L Hk B -> Dk L (Hk r) B"; r=v_per_k)
    end

    # Initialize state
    S = if isnothing(initial_state)
        zeros(T, Dk, Dv, Hv, B)
    else
        copy(initial_state)
    end

    # Output buffer
    O = similar(v)

    for t in 1:L
        # Slice for this timestep
        qt = q[:, t, :, :]   # (Dk, H, B)
        kt = k[:, t, :, :]   # (Dk, H, B)
        vt = v[:, t, :, :]   # (Dv, H, B)
        βt = beta[:, t, :]   # (H, B)
        gt = gate[:, t, :]   # (H, B)

        # Decay state
        S = S .* exp.(reshape(gt, 1, 1, Hv, B))

        # Delta rule: delta = β * (v - Sᵀk)
        sᵀk = einsum(S, kt, einops"Dk Dv h B, Dk h B -> 1 Dv h B")
        delta = reshape(βt, 1, 1, Hv, B) .* (reshape(vt, 1, Dv, Hv, B) .- sᵀk)

        # State update: S += k ⊗ delta
        S = S .+ reshape(kt, Dk, 1, Hv, B) .* delta

        # Output: o = Sᵀq
        O[:, t, :, :] .= einsum(S, qt, einops"Dk Dv h B, Dk h B -> Dv h B")
    end

    return O, S
end
