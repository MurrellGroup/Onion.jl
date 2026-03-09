"""
    block_sparse_proj_dispatch(i, H, tile_m) -> NamedTuple

Build dispatch tables for block-sparse projection execution.
Sorts `(slot, token)` pairs by head, pads to `tile_m` alignment.

Returns:
- `sorted_ids::Vector{Int32}` — `(padded_M,)` flat `(tok-1)*k + slot` indices, padded with `k*L+1`
- `tok_cols::Vector{Int32}` — `(padded_M,)` token column (L-space) per entry, padded with `1`
- `block_heads::Vector{Int32}` — `(num_blocks,)` head id per `tile_m` block
- `head_block_starts::Vector{Int32}` — `(H+1,)` block range: h's blocks = `starts[h]:starts[h+1]-1`
- `slot_dispatches::Vector{NamedTuple}` — per-slot dispatch for accumulation kernels
- `padded_M::Int32`, `k::Int32`, `H::Int32`
"""
function block_sparse_proj_dispatch(i::AbstractMatrix{<:Integer}, H::Integer, tile_m::Integer)
    i_cpu = Int32.(Array(i))
    k, L = size(i_cpu)

    # Padding sentinels: point to an extra "trash" column in the output.
    # Gathers from input use index 1 (valid, result discarded to trash column).
    # Scatters go to the trash column (output_size + 1), which callers allocate.
    flat_trash = Int32(k * L + 1)   # trash column in Y (E, k*L+1)
    tok_safe   = Int32(1)           # valid gather index in X (D, L)

    # ── Flat dispatch (for in_proj fwd, out_proj bwd_dy) ─────────────
    per_head = [Int32[] for _ in 1:H]
    for tok in 1:L, slot in 1:k
        head = Int(i_cpu[slot, tok])
        push!(per_head[head], Int32((tok - 1) * k + slot))
    end

    sorted_ids = Int32[]
    tok_cols = Int32[]
    block_heads = Int32[]
    head_block_starts = Vector{Int32}(undef, H + 1)
    next_block = Int32(1)

    for head in 1:H
        head_block_starts[head] = next_block
        ids = per_head[head]
        toks = @. Int32(fld(ids - Int32(1), Int32(k)) + Int32(1))
        append!(sorted_ids, ids)
        append!(tok_cols, toks)

        num_blocks = cld(length(ids), tile_m)
        pad = num_blocks * tile_m - length(ids)
        append!(sorted_ids, fill(flat_trash, pad))
        append!(tok_cols, fill(tok_safe, pad))
        append!(block_heads, fill(Int32(head), num_blocks))
        next_block += Int32(num_blocks)
    end
    head_block_starts[H + 1] = next_block

    # ── Per-slot dispatches (for out_proj fwd, in_proj bwd_dx) ───────
    slot_dispatches = _build_slot_dispatches(i_cpu, H, k, L, tile_m)

    return (;
        sorted_ids, tok_cols, block_heads, head_block_starts,
        slot_dispatches,
        padded_M = Int32(length(sorted_ids)),
        k = Int32(k),
        H = Int32(H),
    )
end

function _build_slot_dispatches(i_cpu, H, k, L, tile_m)
    tok_trash  = Int32(L + 1)   # trash column in Z (D, L+1)
    flat_safe  = Int32(1)       # valid gather index in Y (E, k*L)

    dispatches = Vector{NamedTuple}(undef, k)
    for slot in 1:k
        per_head = [Int32[] for _ in 1:H]
        for tok in 1:L
            push!(per_head[Int(i_cpu[slot, tok])], Int32(tok))
        end

        sorted_tok_ids = Int32[]
        sorted_flat_ids = Int32[]
        slot_block_heads = Int32[]
        slot_head_block_starts = Vector{Int32}(undef, H + 1)
        next_block = Int32(1)

        for head in 1:H
            slot_head_block_starts[head] = next_block
            tids = per_head[head]
            fids = @. Int32((tids - Int32(1)) * Int32(k) + Int32(slot))
            append!(sorted_tok_ids, tids)
            append!(sorted_flat_ids, fids)

            num_blocks = cld(length(tids), tile_m)
            pad = num_blocks * tile_m - length(tids)
            append!(sorted_tok_ids, fill(tok_trash, pad))
            append!(sorted_flat_ids, fill(flat_safe, pad))
            append!(slot_block_heads, fill(Int32(head), num_blocks))
            next_block += Int32(num_blocks)
        end
        slot_head_block_starts[H + 1] = next_block

        dispatches[slot] = (;
            sorted_tok_ids, sorted_flat_ids,
            block_heads = slot_block_heads,
            head_block_starts = slot_head_block_starts,
            padded_L = Int32(length(sorted_tok_ids)),
        )
    end
    return dispatches
end
