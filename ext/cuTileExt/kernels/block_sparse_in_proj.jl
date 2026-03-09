# ── Forward kernel ────────────────────────────────────────────────────
# Y[:, flat_id] = W_in[:, :, head] @ X[:, tok]
#
# Grid: 1D with 2D swizzle over (sorted_M_blocks × E_blocks)
# All TILE_M entries in a block share the same head → single weight load.
# Dispatch pads with 0; the kernel masks padding via scatter to index 0.

function block_sparse_in_proj_fwd(
    W::TileArray3,                       # (E, D, H)
    X::TileMatrix,                       # (D, L)
    Y::TileMatrix,                       # (E, k*L)
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
    E_dim = size(W, 1)

    bid_m_0, bid_e_0 = swizzle_2d(padded_M, E_dim, TILE_M, TILE_E, GROUP_SIZE_M, bid - 1i32)
    bid_m = bid_m_0 + 1i32
    bid_e = bid_e_0 + 1i32

    fids = ct.load(sorted_ids, (bid_m,), (TILE_M,); padding_mode)
    toks = ct.load(tok_cols, (bid_m,), (TILE_M,); padding_mode)
    head = block_heads[bid_m]

    acc = ct.zeros((TILE_E, TILE_M), Tacc)
    num_d = cld(size(X, 1), Int32(TILE_D))

    d = 1i32
    while d <= num_d
        w_tile = dropdims(ct.load(W, (bid_e, d, head), (TILE_E, TILE_D, 1); padding_mode), dims=3)
        x_tile = ct.gather(X, (reshape((d - 1i32) * Int32(TILE_D) .+ ct.arange((TILE_D,), Int32), (TILE_D, 1)),
                                reshape(toks, (1, TILE_M))))
        acc = muladd(w_tile → Tc, x_tile → Tc, acc)
        d += 1i32
    end

    e_range = (bid_e_0) * Int32(TILE_E) .+ ct.arange((TILE_E,), Int32)
    ct.scatter(Y, (reshape(e_range, (TILE_E, 1)),
                    reshape(fids, (1, TILE_M))),
               acc → eltype(Y))
    return
end

# ── Backward dX kernel (per-slot dispatch + atomic) ──────────────────
# dX[:, tok] += W_in[:, :, head]ᵀ @ dY[:, flat_id]
#
# Called once per slot with per-slot dispatch arrays.
# Each slot has unique tok entries → zero contention on atomic_add.
# Padding entries gather from dY's zeroed trash column → add 0.

function block_sparse_in_proj_bwd_dx(
    W::TileArray3,                       # (E, D, H)
    dY::TileMatrix,                      # (E, k*L+1) — +1 zeroed trash column
    dX::TileMatrix,                      # (D, L)
    sorted_ids::TileVector{Int32},       # (padded_M,) — flat kL indices
    tok_cols::TileVector{Int32},         # (padded_M,) — token column indices
    block_heads::TileVector{Int32},      # (num_blocks,)
    Tc::Type, Tacc::Type,
    TILE_D::Int, TILE_E::Int, TILE_M::Int,
    GROUP_SIZE_M::Int,
)
    padding_mode = ct.PaddingMode.Zero
    bid = ct.bid(1)

    padded_M = size(sorted_ids, 1)
    D_dim = size(dX, 1)

    bid_m_0, bid_d_0 = swizzle_2d(padded_M, D_dim, TILE_M, TILE_D, GROUP_SIZE_M, bid - 1i32)
    bid_m = bid_m_0 + 1i32
    bid_d = bid_d_0 + 1i32

    fids = ct.load(sorted_ids, (bid_m,), (TILE_M,); padding_mode)
    toks = ct.load(tok_cols, (bid_m,), (TILE_M,); padding_mode)
    head = block_heads[bid_m]

    d_range = (bid_d_0) * Int32(TILE_D) .+ ct.arange((TILE_D,), Int32)
    d_idx = reshape(d_range, (TILE_D, 1))

    acc = ct.zeros((TILE_D, TILE_M), Tacc)
    num_e = cld(size(dY, 1), Int32(TILE_E))

    e = 1i32
    while e <= num_e
        e_range = (e - 1i32) * Int32(TILE_E) .+ ct.arange((TILE_E,), Int32)
        w_tile = dropdims(ct.load(W, (e, bid_d, head), (TILE_E, TILE_D, 1); padding_mode), dims=3)
        wt = (w_tile)ᵀ
        dy_tile = ct.gather(dY, (reshape(e_range, (TILE_E, 1)),
                                  reshape(fids, (1, TILE_M))))
        acc = muladd(wt → Tc, dy_tile → Tc, acc)
        e += 1i32
    end

    ct.atomic_add(dX, (d_idx, reshape(toks, (1, TILE_M))), acc → eltype(dX))
    return
end

# ── Backward dW kernel ───────────────────────────────────────────────
# dW[:, :, h] += Σ dY[:, flat_id] @ X[:, tok]ᵀ
#
# Grid: (E_blocks, D_blocks, H * num_splits)
# Split blocks per head for parallelism; atomic_add to accumulate.

function block_sparse_in_proj_bwd_dw(
    dY::TileMatrix,                      # (E, k*L)
    X::TileMatrix,                       # (D, L)
    dW::TileArray3,                      # (E, D, H)
    sorted_ids::TileVector{Int32},       # (padded_M,)
    tok_cols::TileVector{Int32},         # (padded_M,)
    head_block_starts::TileVector{Int32},# (H+1,)
    Tc::Type, Tacc::Type,
    TILE_E::Int, TILE_D::Int, TILE_M::Int,
    num_splits::Int,
)
    padding_mode = ct.PaddingMode.Zero
    bid_e = ct.bid(1)
    bid_d = ct.bid(2)
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

    e_range = (bid_e - 1i32) * Int32(TILE_E) .+ ct.arange((TILE_E,), Int32)
    d_range = (bid_d - 1i32) * Int32(TILE_D) .+ ct.arange((TILE_D,), Int32)
    e_idx = reshape(e_range, (TILE_E, 1))
    d_idx = reshape(d_range, (TILE_D, 1))

    acc = ct.zeros((TILE_E, TILE_D), Tacc)

    blk = my_start
    while blk < my_stop
        fids = ct.load(sorted_ids, (blk,), (TILE_M,); padding_mode)
        toks = ct.load(tok_cols, (blk,), (TILE_M,); padding_mode)

        dy_tile = ct.gather(dY, (e_idx, reshape(fids, (1, TILE_M))))
        x_tile  = ct.gather(X,  (d_idx, reshape(toks, (1, TILE_M))))

        acc = muladd(dy_tile → Tc, (x_tile)ᵀ → Tc, acc)
        blk += 1i32
    end

    ct.atomic_add(dW, (bid_e, bid_d, head),
                  reshape(acc, (TILE_E, TILE_D, 1)) → eltype(dW))
    return
end

# ── Launchers ─────────────────────────────────────────────────────────

function block_sparse_in_proj_fwd!(Y, W, X,
    sorted_ids, tok_cols, block_heads;
    TILE_M,
    tensorcore = tensorcore_type(eltype(X)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing,
)
    E = size(W, 1)
    num_blocks = length(block_heads)

    key = (eltype(X), tensorcore, accumulate,
           nextpow.(4, (E, size(X, 1), num_blocks)), TILE_M)

    autotune_launch(block_sparse_in_proj_fwd,
        CartesianSpace(
            TILE_E=(32, 64, 128), TILE_D=(32, 64, 128),
            GROUP_SIZE_M=(8,), occupancy=(1,),
        ),
        cfg -> (num_blocks * cld(E, cfg.TILE_E),),
        cfg -> (
            W, X, Y, sorted_ids, tok_cols, block_heads,
            Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_E), Constant(cfg.TILE_D), Constant(TILE_M),
            Constant(cfg.GROUP_SIZE_M),
        );
        key, verify,
    )
    return nothing
end

function block_sparse_in_proj_bwd_dx!(dX, W, dY,
    sorted_ids, tok_cols, block_heads;
    TILE_M,
    tensorcore = tensorcore_type(eltype(dY)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing,
)
    D = size(dX, 1)
    num_blocks = length(block_heads)

    key = (:bwd_dx, eltype(dY), tensorcore, accumulate,
           nextpow.(4, (D, size(dY, 1), num_blocks)), TILE_M)

    autotune_launch(block_sparse_in_proj_bwd_dx,
        CartesianSpace(
            TILE_D=(32, 64, 128), TILE_E=(32, 64, 128),
            GROUP_SIZE_M=(8,), occupancy=(1,),
        ),
        cfg -> (num_blocks * cld(D, cfg.TILE_D),),
        cfg -> (
            W, dY, dX, sorted_ids, tok_cols, block_heads,
            Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_D), Constant(cfg.TILE_E), Constant(TILE_M),
            Constant(cfg.GROUP_SIZE_M),
        );
        key, verify,
    )
    return nothing
end

function block_sparse_in_proj_bwd_dw!(dW, dY, X,
    sorted_ids, tok_cols, head_block_starts;
    TILE_M, num_splits = 2,
    tensorcore = tensorcore_type(eltype(X)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing,
)
    E, D, H = size(dW)

    key = (:bwd_dw, eltype(X), tensorcore, accumulate,
           nextpow.(4, (E, D)), H, TILE_M)

    autotune_launch(block_sparse_in_proj_bwd_dw,
        CartesianSpace(
            TILE_E=(32, 64, 128), TILE_D=(32, 64, 128),
            occupancy=(1, 2, 4),
        ),
        cfg -> (cld(E, cfg.TILE_E), cld(D, cfg.TILE_D), H * num_splits),
        cfg -> (
            dY, X, dW, sorted_ids, tok_cols, head_block_starts,
            Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_E), Constant(cfg.TILE_D), Constant(TILE_M),
            Constant(num_splits),
        );
        key, verify,
    )
    return nothing
end
