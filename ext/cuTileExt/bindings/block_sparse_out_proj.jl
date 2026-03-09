# ── Forward ──────────────────────────────────────────────────────────

function Onion.block_sparse_out_proj(::cuTileBackend,
    w::AbstractArray{<:Any,3},
    y::AbstractMatrix,
    i::AbstractMatrix{<:Integer},
)
    D, E, H = size(w)
    k, L = size(i)
    gd = _build_gpu_dispatch(y, i, H, BLOCK_SPARSE_TILE_M)

    # Pad y so sorted_flat_ids padding (1) gathers valid data (discarded to trash)
    y_padded = _pad_flat(y, E, k * L)

    # Per-slot dispatch: each slot has unique toks → zero contention on atomic_add
    Z = fill!(similar(y, D, L + 1), 0)   # +1 trash column for scatter
    for sd in gd.slot_dispatches
        block_sparse_out_proj_fwd!(Z, w, y_padded,
            sd.sorted_flat_ids, sd.sorted_tok_ids, sd.block_heads;
            TILE_M = BLOCK_SPARSE_TILE_M)
    end

    return Z[:, 1:L]
end

# ── rrule ────────────────────────────────────────────────────────────

function CRC.rrule(::typeof(Onion.block_sparse_out_proj), ::cuTileBackend,
    w::AbstractArray{<:Any,3},
    y::AbstractMatrix,
    i::AbstractMatrix{<:Integer},
)
    D, E, H = size(w)
    k, L = size(i)
    gd = _build_gpu_dispatch(y, i, H, BLOCK_SPARSE_TILE_M)

    # Pad y so sorted_flat_ids padding (1) gathers valid data (discarded to trash)
    y_padded = _pad_flat(y, E, k * L)

    # Per-slot dispatch: each slot has unique toks → zero contention on atomic_add
    Z = fill!(similar(y, D, L + 1), 0)   # +1 trash column for scatter
    for sd in gd.slot_dispatches
        block_sparse_out_proj_fwd!(Z, w, y_padded,
            sd.sorted_flat_ids, sd.sorted_tok_ids, sd.block_heads;
            TILE_M = BLOCK_SPARSE_TILE_M)
    end

    # gd and y_padded captured → alive for backward
    function out_proj_pullback(Z̄)
        dZ = unthunk(Z̄)

        # dY — flat dispatch, each flat_id written once; +1 trash column for scatter
        dY = fill!(similar(y, E, k * L + 1), 0)
        block_sparse_out_proj_bwd_dy!(dY, w, dZ,
            gd.sorted_ids, gd.tok_cols, gd.block_heads;
            TILE_M = BLOCK_SPARSE_TILE_M)

        # dW — y_padded already has trash column for gather
        dW = fill!(similar(w), 0)
        block_sparse_out_proj_bwd_dw!(dW, dZ, y_padded,
            gd.sorted_ids, gd.tok_cols, gd.head_block_starts;
            TILE_M = BLOCK_SPARSE_TILE_M)

        return NoTangent(), NoTangent(), dW, dY[:, 1:k*L], NoTangent()
    end

    return Z[:, 1:L], out_proj_pullback
end
