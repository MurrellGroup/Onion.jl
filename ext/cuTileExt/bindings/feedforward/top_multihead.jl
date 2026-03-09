# ── Forward ──────────────────────────────────────────────────────────

function _top_mhffn_forward(Q, K, U, V, i;
                            R = nothing, D_E = nothing,
                            TILE_M = BLOCK_SPARSE_TILE_M)
    D, k, L = size(Q)
    _, _, H = size(K)

    d = _block_sparse_dispatch(i, H, TILE_M)
    kL = k * L

    sorted_ids_d        = _to_device(Q, d.sorted_ids)
    sorted_head_ids_d   = _to_device(Q, d.block_heads)
    head_block_starts_d = _to_device(Q, d.head_block_starts)

    Q_flat = reshape(Q, D, kL)
    O_flat = fill!(similar(Q, D, kL), 0)

    top_mhffn_fwd!(O_flat, Q_flat, K, U, V,
                    sorted_ids_d, sorted_head_ids_d;
                    R, D_E, TILE_M)

    cache = (sorted_ids_d, sorted_head_ids_d, head_block_starts_d,
             Q_flat, TILE_M)

    return O_flat, kL, cache
end

# ── Non-expert dispatch ──────────────────────────────────────────────

function Onion.top_multihead_ffn(::cuTileBackend,
    Q::AbstractArray{<:Any,3}, K::AbstractArray{<:Any,3},
    U::AbstractArray{<:Any,3}, V::AbstractArray{<:Any,3},
    ::typeof(Onion.swish), i::AbstractMatrix,
)
    D, k, L = size(Q)
    O_flat, kL, _ = _top_mhffn_forward(Q, K, U, V, i)
    return reshape(O_flat, D, k, L)
end

function CRC.rrule(::typeof(Onion.top_multihead_ffn), ::cuTileBackend,
    Q::AbstractArray{<:Any,3}, K::AbstractArray{<:Any,3},
    U::AbstractArray{<:Any,3}, V::AbstractArray{<:Any,3},
    ::typeof(Onion.swish), i::AbstractMatrix,
)
    D, k, L = size(Q)
    O_flat, kL, cache = _top_mhffn_forward(Q, K, U, V, i)
    sorted_ids_d, sorted_head_ids_d, head_block_starts_d, Q_flat, TILE_M = cache

    Y = reshape(O_flat, D, k, L)

    function top_mhffn_pullback(Ȳ)
        dO = reshape(unthunk(Ȳ), D, kL)

        dQ_flat = fill!(similar(Q_flat), 0)
        top_mhffn_bwd_dq!(dQ_flat, Q_flat, K, U, V, dO,
                           sorted_ids_d, sorted_head_ids_d;
                           TILE_M)

        dK = similar(K)
        dU = similar(U)
        dV = similar(V)
        top_mhffn_bwd_dkuv!(dK, dU, dV, Q_flat, K, U, V, dO,
                             sorted_ids_d, sorted_head_ids_d,
                             head_block_starts_d;
                             TILE_M)

        dQ = reshape(dQ_flat, D, k, L)

        return NoTangent(), NoTangent(), dQ, dK, dU, dV, NoTangent(), NoTangent()
    end

    return Y, top_mhffn_pullback
end

# ── Expert dispatch ──────────────────────────────────────────────────

function Onion.top_multihead_ffn(::cuTileBackend,
    Q::AbstractArray{<:Any,3}, K::AbstractArray{<:Any,3},
    U::AbstractArray{<:Any,3}, V::AbstractArray{<:Any,3},
    ::typeof(Onion.swish), i::AbstractMatrix,
    R::AbstractArray,
)
    D, k, L = size(Q)
    I = size(K, 1)
    E = size(R, 1)
    D_E = I ÷ E
    O_flat, kL, _ = _top_mhffn_forward(Q, K, U, V, i; R = reshape(R, E, k * L), D_E)
    return reshape(O_flat, D, k, L)
end

function CRC.rrule(::typeof(Onion.top_multihead_ffn), ::cuTileBackend,
    Q::AbstractArray{<:Any,3}, K::AbstractArray{<:Any,3},
    U::AbstractArray{<:Any,3}, V::AbstractArray{<:Any,3},
    ::typeof(Onion.swish), i::AbstractMatrix,
    R::AbstractArray,
)
    D, k, L = size(Q)
    I = size(K, 1)
    E = size(R, 1)
    D_E = I ÷ E
    R_flat = reshape(R, E, k * L)

    O_flat, kL, cache = _top_mhffn_forward(Q, K, U, V, i; R = R_flat, D_E)
    sorted_ids_d, sorted_head_ids_d, head_block_starts_d, Q_flat, TILE_M = cache

    Y = reshape(O_flat, D, k, L)

    function top_mhffn_expert_pullback(Ȳ)
        dO = reshape(unthunk(Ȳ), D, kL)

        dQ_flat = fill!(similar(Q_flat), 0)
        R̄_flat = fill!(similar(R_flat), 0)
        top_mhffn_bwd_dq!(dQ_flat, Q_flat, K, U, V, dO,
                           sorted_ids_d, sorted_head_ids_d;
                           R = R_flat, R̄ = R̄_flat, D_E, TILE_M)

        dK = similar(K)
        dU = similar(U)
        dV = similar(V)
        top_mhffn_bwd_dkuv!(dK, dU, dV, Q_flat, K, U, V, dO,
                             sorted_ids_d, sorted_head_ids_d,
                             head_block_starts_d;
                             R = R_flat, D_E, TILE_M)

        dQ = reshape(dQ_flat, D, k, L)
        dR = reshape(R̄_flat, size(R))

        return NoTangent(), NoTangent(), dQ, dK, dU, dV, NoTangent(), NoTangent(), dR
    end

    return Y, top_mhffn_expert_pullback
end
