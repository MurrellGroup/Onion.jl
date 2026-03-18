using Einops: einsum, @einops_str

"""
    deltanet_recurrent_decode(q, k, v, beta, gate, state) -> (output, state)

Gated DeltaNet recurrent step (decode). Always batched: q/k/v are 3D `(D, H, B)`.
Updates state in-place:
    S = exp(gate) * S
    delta = beta * (v - Sᵀk)
    S += k ⊗ delta
    output = Sᵀq
"""
@primitive deltanet_recurrent_decode
@primitive deltanet_recurrent_decode!

function deltanet_recurrent_decode!(::DefaultBackend,
    output::AbstractArray{T},
    q::AbstractArray{T,3}, k::AbstractArray{T,3}, v::AbstractArray{T,3},
    beta::AbstractMatrix{T}, gate::AbstractMatrix{T},
    state::AbstractArray{T,4}
) where T
    result, state = deltanet_recurrent_decode(DefaultBackend(), q, k, v, beta, gate, state)
    copyto!(output, result)
    return output, state
end

get_buffers(::typeof(deltanet_recurrent_decode), b::Backend, q, k, v, beta, gate, state) =
    (; output = similar(v))

function deltanet_recurrent_decode(::DefaultBackend,
    q::AbstractArray{T,3},    # (Dk, Hk, B)
    k::AbstractArray{T,3},    # (Dk, Hk, B)
    v::AbstractArray{T,3},    # (Dv, Hv, B)
    beta::AbstractMatrix{T},  # (Hv, B)
    gate::AbstractMatrix{T},  # (Hv, B)
    state::AbstractArray{T,4} # (Dk, Dv, H, B) — mutated in-place
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

    output = einsum(state, q, einops"Dk Dv h ..., Dk h ... -> Dv h ...")

    return output, state
end
