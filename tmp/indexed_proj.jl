const INDEXED_PROJ_TILE_M = 64

_to_device(ref, v) = copyto!(similar(ref, eltype(v), size(v)), v)

# ── Dispatch core ────────────────────────────────────────────────────
#
# Download i to CPU, sort + layout + pad there, upload final arrays.
# Kernels derive token_cols and w_offsets internally from sorted_ids,
# so we only upload: sorted_ids, sorted_head_ids, head_block_starts.

function _dispatch_layout(i, H, tile_m)
    k, L = size(i)
    kL = k * L

    # Single download
    head_ids = Int32.(vec(Array(i)))
    perm = sortperm(head_ids)

    # Pass 1: histogram + layout from sorted order
    counts = zeros(Int32, H)
    @inbounds for j in 1:kL
        counts[head_ids[perm[j]]] += Int32(1)
    end

    tm = Int32(tile_m)
    num_blk = cld.(counts, tm)
    total_padded = sum(num_blk .* tm)
    total_blocks = sum(num_blk)

    # Destination offsets for each head in padded output
    dst_starts = Vector{Int32}(undef, H)
    pos = Int32(1)
    @inbounds for h in 1:H
        dst_starts[h] = pos
        pos += num_blk[h] * tm
    end

    # Pass 2: scatter perm into padded layout
    padded_sorted_ids = zeros(Int32, total_padded)
    write_pos = copy(dst_starts)
    @inbounds for j in 1:kL
        idx = perm[j]
        h = head_ids[idx]
        padded_sorted_ids[write_pos[h]] = Int32(idx)
        write_pos[h] += Int32(1)
    end

    # Per-block metadata
    block_prefix = Int32[0; cumsum(num_blk)]
    sorted_head_ids = Vector{Int32}(undef, total_blocks)
    head_block_starts = Vector{Int32}(undef, H + 1)
    @inbounds for h in 1:H
        head_block_starts[h] = block_prefix[h] + Int32(1)
        for b in (block_prefix[h]+1):block_prefix[h+1]
            sorted_head_ids[b] = Int32(h)
        end
    end
    head_block_starts[H + 1] = total_blocks + Int32(1)

    return (; padded_sorted_ids, sorted_head_ids, head_block_starts)
end

# ── Shared binding dispatch ──────────────────────────────────────────

function _indexed_proj_dispatch(W, i, TILE_M)
    d = _dispatch_layout(i, size(W, 3), TILE_M)

    # Upload 3 arrays (3 GPU allocs) — kernels derive token_cols and w_offsets internally
    return (; sorted_ids_d       = _to_device(W, d.padded_sorted_ids),
              sorted_head_ids_d  = _to_device(W, d.sorted_head_ids),
              head_block_starts_d = _to_device(W, d.head_block_starts))
end

# ── scatter_proj ─────────────────────────────────────────────────────
#
# W[E, D, H]  x[D, L]  i[k, L]  →  y[E, k, L]
# mode=0: scatter to kL (flat_ids), gather from L (token_cols)

function Onion.scatter_proj(::cuTileBackend,
    w::AbstractArray{<:Any,3}, x::AbstractMatrix, i::AbstractMatrix,
)
    E, _, _ = size(w)
    k, L = size(i)
    TILE_M = INDEXED_PROJ_TILE_M

    d = _indexed_proj_dispatch(w, i, TILE_M)

    Y_flat = fill!(similar(x, E, k * L), 0)
    indexed_proj!(w, x, Y_flat,
                  d.sorted_ids_d, d.sorted_head_ids_d;
                  TILE_M, scatter_mode=0, gather_mode=1, k_slots=k)

    return reshape(Y_flat, E, k, L)
end

function CRC.rrule(::typeof(Onion.scatter_proj), ::cuTileBackend,
    w::AbstractArray{<:Any,3}, x::AbstractMatrix, i::AbstractMatrix,
)
    E, _, _ = size(w)
    k, L = size(i)
    TILE_M = INDEXED_PROJ_TILE_M

    d = _indexed_proj_dispatch(w, i, TILE_M)

    Y_flat = fill!(similar(x, E, k * L), 0)
    indexed_proj!(w, x, Y_flat,
                  d.sorted_ids_d, d.sorted_head_ids_d;
                  TILE_M, scatter_mode=0, gather_mode=1, k_slots=k)

    Y = reshape(Y_flat, E, k, L)

    function scatter_proj_pullback(Ȳ)
        dY_flat = reshape(unthunk(Ȳ), E, k * L)

        # dX: gather from dY (kL), scatter-add to dX (L)
        dX = fill!(similar(x), 0)
        ∇indexed_proj_dx!(w, dY_flat, dX,
                           d.sorted_ids_d, d.sorted_head_ids_d;
                           TILE_M, mode=0, k_slots=k)

        # dW
        dW = similar(w)
        ∇indexed_proj_dw!(x, dY_flat, dW,
                           d.sorted_ids_d, d.head_block_starts_d;
                           TILE_M, k_slots=k, mode=0)

        return NoTangent(), NoTangent(), dW, dX, NoTangent()
    end

    return Y, scatter_proj_pullback
end

# ── gather_proj ──────────────────────────────────────────────────────
#
# W[D, E, H]  z[E, k, L]  i[k, L]  →  o[D, L]
# scatter_mode=1, gather_mode=0: gather from flat_ids, atomic_add to tok_cols.
# Scatters directly to (D, L) — multiple k-slots per token accumulate via atomics.

function Onion.gather_proj(::cuTileBackend,
    w::AbstractArray{<:Any,3}, z::AbstractArray{<:Any,3}, i::AbstractMatrix,
)
    D, _, _ = size(w)
    k, L = size(i)
    TILE_M = INDEXED_PROJ_TILE_M

    d = _indexed_proj_dispatch(w, i, TILE_M)
    Z_flat = reshape(z, size(z, 1), k * L)

    O = fill!(similar(z, D, L), 0)
    indexed_proj!(w, Z_flat, O,
                  d.sorted_ids_d, d.sorted_head_ids_d;
                  TILE_M, scatter_mode=1, gather_mode=0, k_slots=k,
                  use_atomic=1)

    return O
end

function CRC.rrule(::typeof(Onion.gather_proj), ::cuTileBackend,
    w::AbstractArray{<:Any,3}, z::AbstractArray{<:Any,3}, i::AbstractMatrix,
)
    D, E, _ = size(w)
    k, L = size(i)
    TILE_M = INDEXED_PROJ_TILE_M

    d = _indexed_proj_dispatch(w, i, TILE_M)
    Z_flat = reshape(z, E, k * L)

    O = fill!(similar(z, D, L), 0)
    indexed_proj!(w, Z_flat, O,
                  d.sorted_ids_d, d.sorted_head_ids_d;
                  TILE_M, scatter_mode=1, gather_mode=0, k_slots=k,
                  use_atomic=1)

    function gather_proj_pullback(Ō)
        dO = unthunk(Ō)

        # dZ: gather from dO (L), scatter to dZ (kL)
        dZ_flat = fill!(similar(z, E, k * L), 0)
        ∇indexed_proj_dx!(w, dO, dZ_flat,
                           d.sorted_ids_d, d.sorted_head_ids_d;
                           TILE_M, mode=1, k_slots=k)
        dZ = reshape(dZ_flat, E, k, L)

        # dW
        dW = similar(w)
        ∇indexed_proj_dw!(Z_flat, dO, dW,
                           d.sorted_ids_d, d.head_block_starts_d;
                           TILE_M, k_slots=k, mode=1)

        return NoTangent(), NoTangent(), dW, dZ, NoTangent()
    end

    return O, gather_proj_pullback
end
