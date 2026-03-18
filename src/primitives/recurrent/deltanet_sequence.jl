using Einops: einsum, @einops_str

"""
    deltanet_sequence(q, k, v, beta, gate, initial_state=nothing) -> (output, final_state)

Full-sequence gated DeltaNet recurrence. Always batched:

    q, k:          (Dk, T, H, B)
    v:             (Dv, T, H, B)
    beta, gate:    (H, T, B)
    initial_state: (Dk, Dv, H, B) or nothing
    output:        (Dv, T, H, B)
    final_state:   (Dk, Dv, H, B)
"""
@primitive deltanet_sequence
@primitive deltanet_sequence!

function get_buffers(::typeof(deltanet_sequence), b::Backend,
    q::AbstractArray{T,4}, k, v::AbstractArray{T,4}, beta, gate,
    initial_state=nothing
) where T
    Dk = size(q, 1)
    Dv, _, Hv, B = size(v)
    (; output = similar(v), final_state = similar(v, T, Dk, Dv, Hv, B))
end

function deltanet_sequence(b::Backend,
    q::AbstractArray{T,4}, k::AbstractArray{T,4}, v::AbstractArray{T,4},
    beta::AbstractArray{T,3}, gate::AbstractArray{T,3},
    initial_state::Optional{AbstractArray{T,4}} = nothing,
) where T
    bufs = get_buffers(deltanet_sequence, b, q, k, v, beta, gate, initial_state)
    deltanet_sequence!(b, bufs.output, bufs.final_state, q, k, v, beta, gate, initial_state)
    return bufs.output, bufs.final_state
end

# Naive sequential recurrence. Correct, simple, AD-friendly (no mutation).
function deltanet_sequence(::DefaultBackend,
    q::AbstractArray{T,4},    # (Dk, L, Hk, B)
    k::AbstractArray{T,4},    # (Dk, L, Hk, B)
    v::AbstractArray{T,4},    # (Dv, L, Hv, B)
    beta::AbstractArray{T,3}, # (Hv, L, B)
    gate::AbstractArray{T,3}, # (Hv, L, B)
    initial_state::Optional{AbstractArray{T,4}} = nothing, # (Dk, Dv, Hv, B)
) where T
    Dk, L, Hk, B = size(q)
    Dv, _, Hv, _ = size(v)

    v_per_k = cld(Hv, Hk)
    if v_per_k > 1
        q = repeat(q, einops"Dk L Hk B -> Dk L (Hk r) B"; r=v_per_k)
        k = repeat(k, einops"Dk L Hk B -> Dk L (Hk r) B"; r=v_per_k)
    end

    S = if isnothing(initial_state)
        zeros(T, Dk, Dv, Hv, B)
    else
        copy(initial_state)
    end

    O = similar(v)

    for t in 1:L
        qt = q[:, t, :, :]
        kt = k[:, t, :, :]
        vt = v[:, t, :, :]
        βt = beta[:, t, :]
        gt = gate[:, t, :]

        S = S .* exp.(reshape(gt, 1, 1, Hv, B))

        sᵀk = einsum(S, kt, einops"Dk Dv h B, Dk h B -> 1 Dv h B")
        delta = reshape(βt, 1, 1, Hv, B) .* (reshape(vt, 1, Dv, Hv, B) .- sᵀk)

        S = S .+ reshape(kt, Dk, 1, Hv, B) .* delta

        O[:, t, :, :] .= einsum(S, qt, einops"Dk Dv h B, Dk h B -> Dv h B")
    end

    return O, S
end
