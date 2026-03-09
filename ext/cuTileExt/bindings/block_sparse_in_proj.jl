const BLOCK_SPARSE_TILE_M = 64

function _to_device(ref, v::AbstractVector)
    copyto!(similar(ref, eltype(v), size(v)), v)
end

# Per-slot dispatch arrays on GPU.
struct _BSPSlotGPU{V<:AbstractVector{Int32}}
    sorted_tok_ids::V    # (padded_L_slot,) — token columns, padding = L+1
    sorted_flat_ids::V   # (padded_L_slot,) — flat Y columns, padding = 1
    block_heads::V       # (num_slot_blocks,)
end

# Bundle all GPU dispatch arrays so lifetimes are tied to a single object.
# Captured by rrule closures → alive for the entire fwd+bwd lifetime.
struct _BSPDispatchGPU{V<:AbstractVector{Int32}}
    sorted_ids::V          # (padded_M,) — flat kL indices, padding = k*L+1
    tok_cols::V            # (padded_M,) — token columns, padding = 1
    block_heads::V         # (num_flat_blocks,)
    head_block_starts::V   # (H+1,)
    slot_dispatches::Vector{_BSPSlotGPU{V}}
    k::Int32
    L::Int32
end

function _build_gpu_dispatch(ref, i, H, tile_m)
    d = Onion.block_sparse_proj_dispatch(i, H, tile_m)
    dev = v -> _to_device(ref, v)
    slots = [_BSPSlotGPU(dev(sd.sorted_tok_ids), dev(sd.sorted_flat_ids),
                         dev(sd.block_heads))
             for sd in d.slot_dispatches]
    _BSPDispatchGPU(dev(d.sorted_ids), dev(d.tok_cols), dev(d.block_heads),
                    dev(d.head_block_starts), slots, d.k, Int32(size(i, 2)))
end

# Pad array with a zeroed trash column at index kL+1.
# Padding entries in sorted_ids gather from this column → contribute 0.
function _pad_flat(x, E, kL)
    xp = fill!(similar(x, E, kL + 1), 0)
    xp[:, 1:kL] .= x
    return xp
end

# ── Forward ──────────────────────────────────────────────────────────

function Onion.block_sparse_in_proj(::cuTileBackend,
    w::AbstractArray{<:Any,3},
    x::AbstractMatrix,
    i::AbstractMatrix{<:Integer},
)
    E, D, H = size(w)
    k, L = size(i)
    gd = _build_gpu_dispatch(x, i, H, BLOCK_SPARSE_TILE_M)

    Y = fill!(similar(x, E, k * L + 1), 0)   # +1 trash column for scatter
    block_sparse_in_proj_fwd!(Y, w, x,
        gd.sorted_ids, gd.tok_cols, gd.block_heads;
        TILE_M = BLOCK_SPARSE_TILE_M)

    return Y[:, 1:k*L]
end

# ── rrule ────────────────────────────────────────────────────────────

function CRC.rrule(::typeof(Onion.block_sparse_in_proj), ::cuTileBackend,
    w::AbstractArray{<:Any,3},
    x::AbstractMatrix,
    i::AbstractMatrix{<:Integer},
)
    E, D, H = size(w)
    k, L = size(i)
    gd = _build_gpu_dispatch(x, i, H, BLOCK_SPARSE_TILE_M)

    Y = fill!(similar(x, E, k * L + 1), 0)   # +1 trash column for scatter
    block_sparse_in_proj_fwd!(Y, w, x,
        gd.sorted_ids, gd.tok_cols, gd.block_heads;
        TILE_M = BLOCK_SPARSE_TILE_M)

    # gd is captured by the closure → all GPU dispatch arrays stay alive
    function in_proj_pullback(Ȳ)
        dY_raw = unthunk(Ȳ)

        # Pad dY so sorted_ids padding (k*L+1) gathers zeros
        dY_padded = _pad_flat(dY_raw, E, k * L)

        # dX — per-slot dispatch, each slot has unique toks → zero contention
        dX = fill!(similar(x, D, L + 1), 0)   # +1 trash column for scatter
        for sd in gd.slot_dispatches
            block_sparse_in_proj_bwd_dx!(dX, w, dY_padded,
                sd.sorted_flat_ids, sd.sorted_tok_ids, sd.block_heads;
                TILE_M = BLOCK_SPARSE_TILE_M)
        end

        # dW
        dW = fill!(similar(w), 0)
        block_sparse_in_proj_bwd_dw!(dW, dY_padded, x,
            gd.sorted_ids, gd.tok_cols, gd.head_block_starts;
            TILE_M = BLOCK_SPARSE_TILE_M)

        return NoTangent(), NoTangent(), dW, dX[:, 1:L], NoTangent()
    end

    return Y[:, 1:k*L], in_proj_pullback
end
