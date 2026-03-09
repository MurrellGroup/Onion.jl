"""
    indexed_proj_dispatch(i, H, tile_m) -> (; sorted_ids, sorted_head_ids, head_block_starts)

Build dispatch table for sorted indexed-projection execution.
Sorts `(k_slot, token)` pairs by head, pads to `tile_m` alignment.
Shared by scatter_proj, gather_proj, and top_multihead_ffn.

Returns a named tuple:
- `sorted_ids::Vector{Int32}` — `(padded_M,)` original flat indices sorted by head, 0-padded
- `sorted_head_ids::Vector{Int32}` — `(num_blocks,)` head id per `tile_m` block
- `head_block_starts::Vector{Int32}` — `(H+1,)` block range per head: h's blocks = `starts[h]:starts[h+1]-1`
"""
function indexed_proj_dispatch(i::AbstractMatrix{<:Integer}, H::Integer, tile_m::Integer)
    k, L = size(i)
    head_ids = vec(Array(i))  # (k*L,) — ensure CPU

    perm = sortperm(head_ids)
    sorted_heads = head_ids[perm]

    sorted_ids = Int32[]
    sorted_head_ids = Int32[]
    head_block_starts = Vector{Int32}(undef, H + 1)

    block_count = Int32(0)

    for h in 1:H
        head_block_starts[h] = block_count + Int32(1)

        first = searchsortedfirst(sorted_heads, h)
        last = searchsortedlast(sorted_heads, h)
        count = last - first + 1

        if count <= 0
            continue
        end

        num_blocks = cld(count, tile_m)

        for idx in first:last
            push!(sorted_ids, Int32(perm[idx]))
        end
        # Pad with sentinel (0 = OOB in 1-based indexing)
        for _ in 1:(num_blocks * tile_m - count)
            push!(sorted_ids, Int32(0))
        end

        for _ in 1:num_blocks
            push!(sorted_head_ids, Int32(h))
        end

        block_count += Int32(num_blocks)
    end
    head_block_starts[H + 1] = block_count + Int32(1)

    return (; sorted_ids, sorted_head_ids, head_block_starts)
end
