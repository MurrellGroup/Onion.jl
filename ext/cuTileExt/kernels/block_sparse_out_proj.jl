# ── Forward kernel (atomic add) ───────────────────────────────────────
# Z[:, tok] += W_out[:, :, head] @ Y[:, flat_id]
#
# Used with per-slot dispatch: each slot has unique tok entries → zero
# contention on atomic_add. Padding gathers from Y's trash column (zeros)
# and scatters to Z's trash column (discarded).

function block_sparse_out_proj_fwd(
    W::TileArray3,                       # (D, E, H)
    Y::TileMatrix,                       # (E, k*L+1) — +1 zeroed trash column
    Z::TileMatrix,                       # (D, L+1)  — +1 trash column for scatter
    sorted_ids::TileVector{Int32},       # (padded_L_slot,) — flat kL indices
    tok_cols::TileVector{Int32},         # (padded_L_slot,) — token column indices
    block_heads::TileVector{Int32},      # (num_blocks,)
    Tc::Type, Tacc::Type,
    TILE_D::Int, TILE_E::Int, TILE_M::Int,
    GROUP_SIZE_M::Int,
)
    padding_mode = ct.PaddingMode.Zero
    bid = ct.bid(1)

    padded_M = size(sorted_ids, 1)
    D_dim = size(W, 1)

    bid_m_0, bid_d_0 = swizzle_2d(padded_M, D_dim, TILE_M, TILE_D, GROUP_SIZE_M, bid - 1i32)
    bid_m = bid_m_0 + 1i32
    bid_d = bid_d_0 + 1i32

    fids = ct.load(sorted_ids, (bid_m,), (TILE_M,); padding_mode)
    toks = ct.load(tok_cols, (bid_m,), (TILE_M,); padding_mode)
    head = block_heads[bid_m]

    d_range = (bid_d_0) * Int32(TILE_D) .+ ct.arange((TILE_D,), Int32)
    d_idx = reshape(d_range, (TILE_D, 1))

    acc = ct.zeros((TILE_D, TILE_M), Tacc)
    num_e = cld(size(Y, 1), Int32(TILE_E))

    e = 1i32
    while e <= num_e
        e_range = (e - 1i32) * Int32(TILE_E) .+ ct.arange((TILE_E,), Int32)
        w_tile = dropdims(ct.load(W, (bid_d, e, head), (TILE_D, TILE_E, 1); padding_mode), dims=3)
        y_tile = ct.gather(Y, (reshape(e_range, (TILE_E, 1)),
                                reshape(fids, (1, TILE_M))))
        acc = muladd(w_tile → Tc, y_tile → Tc, acc)
        e += 1i32
    end

    ct.atomic_add(Z, (d_idx, reshape(toks, (1, TILE_M))), acc → eltype(Z))
    return
end

# ── Backward dY kernel ───────────────────────────────────────────────
# dY[:, flat_id] = W_out[:, :, head]ᵀ @ dZ[:, tok]
#
# Uses flat dispatch — each flat_id is written once, no atomics.

function block_sparse_out_proj_bwd_dy(
    W::TileArray3,                       # (D, E, H)
    dZ::TileMatrix,                      # (D, L)
    dY::TileMatrix,                      # (E, k*L+1)
    sorted_ids::TileVector{Int32},       # (padded_M,) — flat kL indices
    tok_cols::TileVector{Int32},         # (padded_M,) — token column indices
    block_heads::TileVector{Int32},      # (num_blocks,)
    Tc::Type, Tacc::Type,
    TILE_E::Int, TILE_D::Int, TILE_M::Int,
    GROUP_SIZE_M::Int,
)
    padding_mode = ct.PaddingMode.Zero
    bid = ct.bid(1)

    padded_M = size(sorted_ids, 1)
    E_dim = size(dY, 1)

    bid_m_0, bid_e_0 = swizzle_2d(padded_M, E_dim, TILE_M, TILE_E, GROUP_SIZE_M, bid - 1i32)
    bid_m = bid_m_0 + 1i32
    bid_e = bid_e_0 + 1i32

    fids = ct.load(sorted_ids, (bid_m,), (TILE_M,); padding_mode)
    toks = ct.load(tok_cols, (bid_m,), (TILE_M,); padding_mode)
    head = block_heads[bid_m]

    acc = ct.zeros((TILE_E, TILE_M), Tacc)
    num_d = cld(size(dZ, 1), Int32(TILE_D))

    d = 1i32
    while d <= num_d
        d_range = (d - 1i32) * Int32(TILE_D) .+ ct.arange((TILE_D,), Int32)
        w_tile = dropdims(ct.load(W, (d, bid_e, head), (TILE_D, TILE_E, 1); padding_mode), dims=3)
        wt = (w_tile)ᵀ
        dz_tile = ct.gather(dZ, (reshape(d_range, (TILE_D, 1)),
                                  reshape(toks, (1, TILE_M))))
        acc = muladd(wt → Tc, dz_tile → Tc, acc)
        d += 1i32
    end

    e_range = (bid_e_0) * Int32(TILE_E) .+ ct.arange((TILE_E,), Int32)
    ct.scatter(dY, (reshape(e_range, (TILE_E, 1)),
                     reshape(fids, (1, TILE_M))),
               acc → eltype(dY))
    return
end

# ── Backward dW kernel ───────────────────────────────────────────────
# dW[:, :, h] += Σ dZ[:, tok] @ Y[:, flat_id]ᵀ
#
# Grid: (D_blocks, E_blocks, H * num_splits)
# Split blocks per head for parallelism; atomic_add to accumulate.

function block_sparse_out_proj_bwd_dw(
    dZ::TileMatrix,                      # (D, L)
    Y::TileMatrix,                       # (E, k*L+1)
    dW::TileArray3,                      # (D, E, H)
    sorted_ids::TileVector{Int32},       # (padded_M,)
    tok_cols::TileVector{Int32},         # (padded_M,)
    head_block_starts::TileVector{Int32},# (H+1,)
    Tc::Type, Tacc::Type,
    TILE_D::Int, TILE_E::Int, TILE_M::Int,
    num_splits::Int,
)
    padding_mode = ct.PaddingMode.Zero
    bid_d = ct.bid(1)
    bid_e = ct.bid(2)
    bid_hs = ct.bid(3)
    head  = fld(bid_hs - 1i32, Int32(num_splits)) + 1i32
    split = rem(bid_hs - 1i32, Int32(num_splits)) + 1i32

    blk_start = head_block_starts[head]
    blk_stop  = head_block_starts[head + 1i32]
    total = blk_stop - blk_start
    rel_start = fld((split - 1i32) * total, Int32(num_splits))
    rel_stop  = fld(split * total, Int32(num_splits))
    my_start = blk_start + rel_start
    my_stop  = blk_start + rel_stop

    d_range = (bid_d - 1i32) * Int32(TILE_D) .+ ct.arange((TILE_D,), Int32)
    e_range = (bid_e - 1i32) * Int32(TILE_E) .+ ct.arange((TILE_E,), Int32)
    d_idx = reshape(d_range, (TILE_D, 1))
    e_idx = reshape(e_range, (TILE_E, 1))

    acc = ct.zeros((TILE_D, TILE_E), Tacc)

    blk = my_start
    while blk < my_stop
        fids = ct.load(sorted_ids, (blk,), (TILE_M,); padding_mode)
        toks = ct.load(tok_cols, (blk,), (TILE_M,); padding_mode)

        dz_tile = ct.gather(dZ, (d_idx, reshape(toks, (1, TILE_M))))
        y_tile  = ct.gather(Y,  (e_idx, reshape(fids, (1, TILE_M))))

        acc = muladd(dz_tile → Tc, (y_tile)ᵀ → Tc, acc)
        blk += 1i32
    end

    ct.atomic_add(dW, (bid_d, bid_e, head),
                  reshape(acc, (TILE_D, TILE_E, 1)) → eltype(dW))
    return
end

# ── Launchers ─────────────────────────────────────────────────────────

function block_sparse_out_proj_fwd!(Z, W, Y,
    sorted_ids, tok_cols, block_heads;
    TILE_M,
    tensorcore = tensorcore_type(eltype(Y)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing,
)
    D = size(W, 1)
    num_blocks = length(block_heads)

    key = (:out_fwd, eltype(Y), tensorcore, accumulate,
           nextpow.(4, (D, size(Y, 1), num_blocks)), TILE_M)

    autotune_launch(block_sparse_out_proj_fwd,
        CartesianSpace(
            TILE_D=(32, 64, 128), TILE_E=(32, 64, 128),
            GROUP_SIZE_M=(8,), occupancy=(1,),
        ),
        cfg -> (num_blocks * cld(D, cfg.TILE_D),),
        cfg -> (
            W, Y, Z, sorted_ids, tok_cols, block_heads,
            Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_D), Constant(cfg.TILE_E), Constant(TILE_M),
            Constant(cfg.GROUP_SIZE_M),
        );
        key, verify,
    )
    return nothing
end

function block_sparse_out_proj_bwd_dy!(dY, W, dZ,
    sorted_ids, tok_cols, block_heads;
    TILE_M,
    tensorcore = tensorcore_type(eltype(dZ)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing,
)
    E = size(dY, 1)
    num_blocks = length(block_heads)

    key = (:bwd_dy, eltype(dZ), tensorcore, accumulate,
           nextpow.(4, (E, size(dZ, 1), num_blocks)), TILE_M)

    autotune_launch(block_sparse_out_proj_bwd_dy,
        CartesianSpace(
            TILE_E=(32, 64, 128), TILE_D=(32, 64, 128),
            GROUP_SIZE_M=(8,), occupancy=(1,),
        ),
        cfg -> (num_blocks * cld(E, cfg.TILE_E),),
        cfg -> (
            W, dZ, dY, sorted_ids, tok_cols, block_heads,
            Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_E), Constant(cfg.TILE_D), Constant(TILE_M),
            Constant(cfg.GROUP_SIZE_M),
        );
        key, verify,
    )
    return nothing
end

function block_sparse_out_proj_bwd_dw!(dW, dZ, Y,
    sorted_ids, tok_cols, head_block_starts;
    TILE_M, num_splits = 2,
    tensorcore = tensorcore_type(eltype(Y)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing,
)
    D, E, H = size(dW)

    key = (:bwd_dw, eltype(Y), tensorcore, accumulate,
           nextpow.(4, (D, E)), H, TILE_M)

    autotune_launch(block_sparse_out_proj_bwd_dw,
        CartesianSpace(
            TILE_D=(32, 64, 128), TILE_E=(32, 64, 128),
            occupancy=(1, 2, 4),
        ),
        cfg -> (cld(D, cfg.TILE_D), cld(E, cfg.TILE_E), H * num_splits),
        cfg -> (
            dZ, Y, dW, sorted_ids, tok_cols, head_block_starts,
            Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_D), Constant(cfg.TILE_E), Constant(TILE_M),
            Constant(num_splits),
        );
        key, verify,
    )
    return nothing
end
