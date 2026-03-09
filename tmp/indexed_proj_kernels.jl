# ── Forward kernel ────────────────────────────────────────────────────
#
# Y[:, scatter_id] += W[:, :, head] @ X[:, gather_id]
#
# Grid: 1D with 2D swizzle over (sorted_M_blocks × out_blocks)
# All TILE_M tokens in a block share the same head → single weight load.
# Dispatch pads with 0; the kernel masks padding internally.
#
# sorted_ids are flat kL indices. The kernel derives:
#   token_col = cld(sorted_id, k)      — L-space index
#   head      = sorted_head_ids[block] — W slice via ct.load
#
# scatter_mode/gather_mode select index source (0=flat_ids, 1=tok_cols):
#   scatter_proj: scatter_mode=0, gather_mode=1  (scatter to kL, gather from L)
#   gather_proj:  scatter_mode=0, gather_mode=0  (both flat_ids, no atomics)

function indexed_proj_fwd_kernel(
    W::TileMatrix,                        # (out, in*H) — reshaped from (out, in, H)
    X::TileMatrix,                        # (in, gather_cols)
    Y::TileMatrix,                        # (out, scatter_cols)
    sorted_ids::TileVector{Int32},        # (padded_M,) — flat kL indices
    sorted_head_ids::TileVector{Int32},   # (num_blocks,) — head per block
    Tc::Type, Tacc::Type,
    TILE_N::Int, TILE_K::Int, TILE_M::Int,
    GROUP_SIZE_M::Int,
    in_dim::Int,
    k_slots::Int,
    scatter_mode::Int,                    # 0=flat_ids, 1=tok_cols
    gather_mode::Int,                     # 0=flat_ids, 1=tok_cols
    use_atomic::Int,                      # 1=atomic_add scatter, 0=overwrite scatter
)
    padding_mode = ct.PaddingMode.Zero
    bid = ct.bid(1)

    out_dim = size(W, 1)
    padded_M = size(sorted_ids, 1)

    # 2D swizzle over (sorted_M_blocks × out_blocks)
    bid_m_0, bid_n_0 = swizzle_2d(padded_M, out_dim, TILE_M, TILE_N, GROUP_SIZE_M, bid - 1i32)
    bid_m = bid_m_0 + 1i32

    # Load flat kL indices; 0 = padding from dispatch
    fids  = ct.load(sorted_ids, (bid_m,), (TILE_M,); padding_mode)
    valid = fids .>= 1i32
    safe_fids = max.(fids, 1i32)

    # Derive token columns (L-space) from flat kL indices: fld1(id, k)
    tok_cols = (safe_fids .- 1i32) .÷ Int32(k_slots) .+ 1i32

    # W column base offset from per-block head id
    head = sorted_head_ids[bid_m]
    w_base = (head - 1i32) * Int32(in_dim)

    # Select scatter/gather indices based on modes (arithmetic, no branch)
    sm = Int32(scatter_mode); gm = Int32(gather_mode)
    scatter_tgt = sm .* tok_cols .+ (1i32 - sm) .* safe_fids
    scatter_tgt = ifelse.(valid, scatter_tgt, Int32(0))   # OOB → masked by ct.scatter
    gather_src  = gm .* tok_cols .+ (1i32 - gm) .* safe_fids

    # Output row indices (for scatter)
    n_range = bid_n_0 * Int32(TILE_N) .+ ct.arange((TILE_N,), Int32)
    n_idx = reshape(n_range, (TILE_N, 1))

    # K-dimension accumulation loop
    num_k = cld(Int32(in_dim), Int32(TILE_K))
    acc = ct.zeros((TILE_N, TILE_M), Tacc)

    k = 1i32
    while k <= num_k
        k_range = (k - 1i32) * Int32(TILE_K) .+ ct.arange((TILE_K,), Int32)
        k_valid = k_range .<= Int32(in_dim)

        # Gather W tile
        w_col_off = w_base + (k - 1i32) * Int32(TILE_K)
        w_cols = w_col_off .+ ct.arange((TILE_K,), Int32)
        w = ct.gather(W, (n_idx, reshape(w_cols, (1, TILE_K))))
        w = ifelse.(reshape(k_valid, (1, TILE_K)), w, Tacc(0))

        # Gather X columns
        x = ct.gather(X, (reshape(k_range, (TILE_K, 1)),
                          reshape(gather_src, (1, TILE_M))))
        x = ifelse.(reshape(k_valid, (TILE_K, 1)), x, Tacc(0))

        acc = muladd(w → Tc, x → Tc, acc)
        k += 1i32
    end

    # Scatter output — zero padding positions
    acc = ifelse.(reshape(valid, (1, TILE_M)), acc, Tacc(0))
    scatter_idx = (n_idx, reshape(scatter_tgt, (1, TILE_M)))
    if Int32(use_atomic) == 1i32
        ct.atomic_add(Y, scatter_idx, acc → eltype(Y))
    else
        ct.scatter(Y, scatter_idx, acc → eltype(Y))
    end
    return
end

function indexed_proj!(
    W, X, Y,
    sorted_ids, sorted_head_ids;
    TILE_M, scatter_mode, gather_mode, k_slots, use_atomic=0,
    tensorcore = tensorcore_type(eltype(X)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing,
)
    out_dim = size(W, 1)
    in_dim = size(W, 2)
    W_2d = reshape(W, out_dim, :)
    num_sorted_blocks = length(sorted_head_ids)

    key = (eltype(X), tensorcore, accumulate,
           nextpow.(4, (out_dim, in_dim, num_sorted_blocks)),
           TILE_M, scatter_mode, gather_mode, use_atomic)

    autotune_launch(indexed_proj_fwd_kernel,
        CartesianSpace(
            TILE_N=(32, 64, 128), TILE_K=(32, 64, 128),
            GROUP_SIZE_M=(8,), occupancy=(1,),
        ),
        cfg -> (num_sorted_blocks * cld(out_dim, cfg.TILE_N),),
        cfg -> (
            W_2d, X, Y, sorted_ids, sorted_head_ids,
            Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_N), Constant(cfg.TILE_K), Constant(TILE_M),
            Constant(cfg.GROUP_SIZE_M),
            Constant(in_dim),
            Constant(k_slots),
            Constant(scatter_mode), Constant(gather_mode),
            Constant(use_atomic),
        );
        key, verify,
    )
    return nothing
end

# ── Backward dInput kernel ──────────────────────────────────────────
#
# dX[:, scatter_id] += Wᵀ @ dY[:, gather_id]
#
# Same index derivation as forward: sorted_ids are flat kL indices,
# token_col and w_offset derived inside kernel.
# mode: 0 = scatter_proj dX (gather from kL, scatter to L)
#        1 = gather_proj dZ  (gather from L, scatter to kL)

function indexed_proj_bwd_dx_kernel(
    W::TileMatrix,                        # (out, in*H) — reshaped from (out, in, H)
    dY::TileMatrix,                       # (out, dY_cols)
    dX::TileMatrix,                       # (in, dX_cols)
    sorted_ids::TileVector{Int32},        # (padded_M,) — flat kL indices
    sorted_head_ids::TileVector{Int32},   # (num_blocks,) — head per block
    Tc::Type, Tacc::Type,
    TILE_K::Int, TILE_N::Int, TILE_M::Int,
    GROUP_SIZE_M::Int,
    in_dim::Int,
    k_slots::Int,
    mode::Int,
)
    padding_mode = ct.PaddingMode.Zero
    bid = ct.bid(1)

    out_dim = size(W, 1)
    padded_M = size(sorted_ids, 1)

    # 2D swizzle over (sorted_M_blocks × in_blocks)
    bid_m_0, bid_k_0 = swizzle_2d(padded_M, Int32(in_dim), TILE_M, TILE_K, GROUP_SIZE_M, bid - 1i32)
    bid_m = bid_m_0 + 1i32

    # Load flat kL indices
    fids  = ct.load(sorted_ids, (bid_m,), (TILE_M,); padding_mode)
    valid = fids .>= 1i32
    safe_fids = max.(fids, 1i32)

    # Derive token columns and scatter/gather indices
    tok_cols = (safe_fids .- 1i32) .÷ Int32(k_slots) .+ 1i32
    m = Int32(mode)
    inv_m = 1i32 - m
    # mode 0 (scatter_proj dX): gather from kL (flat_ids), scatter to L (tok_cols)
    # mode 1 (gather_proj dZ): gather from L (tok_cols), scatter to kL (flat_ids)
    gather_src  = m .* tok_cols  .+ inv_m .* safe_fids
    scatter_tgt = m .* safe_fids .+ inv_m .* tok_cols
    scatter_tgt = ifelse.(valid, scatter_tgt, Int32(0))   # OOB → masked by ct.scatter

    # Head id → W column base offset
    head = sorted_head_ids[bid_m]
    w_base = (head - 1i32) * Int32(in_dim)

    # W column indices for this k-tile (fixed for block)
    w_k_cols = w_base .+ bid_k_0 * Int32(TILE_K) .+ ct.arange((TILE_K,), Int32)
    k_valid = bid_k_0 * Int32(TILE_K) .+ ct.arange((TILE_K,), Int32) .<= Int32(in_dim)

    # N-dimension loop over out_dim (computing Wᵀ @ dY)
    num_n = cld(out_dim, Int32(TILE_N))
    acc = ct.zeros((TILE_K, TILE_M), Tacc)

    n = 1i32
    while n <= num_n
        n_range = (n - 1i32) * Int32(TILE_N) .+ ct.arange((TILE_N,), Int32)

        # Gather W tile and transpose
        w = ct.gather(W, (reshape(n_range, (TILE_N, 1)),
                          reshape(w_k_cols, (1, TILE_K))))
        w = ifelse.(reshape(k_valid, (1, TILE_K)), w, Tacc(0))
        wt = (w)ᵀ

        # Gather dY columns
        dy = ct.gather(dY, (reshape(n_range, (TILE_N, 1)),
                            reshape(gather_src, (1, TILE_M))))

        acc = muladd(wt → Tc, dy → Tc, acc)
        n += 1i32
    end

    # Scatter dX — zero padding positions
    acc = ifelse.(reshape(valid, (1, TILE_M)), acc, Tacc(0))
    k_range = bid_k_0 * Int32(TILE_K) .+ ct.arange((TILE_K,), Int32)
    ct.scatter(dX, (reshape(k_range, (TILE_K, 1)),
                    reshape(scatter_tgt, (1, TILE_M))),
               acc → eltype(dX))
    return
end

function ∇indexed_proj_dx!(
    W, dY, dX,
    sorted_ids, sorted_head_ids;
    TILE_M, mode, k_slots,
    tensorcore = tensorcore_type(eltype(dY)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing,
)
    out_dim = size(W, 1)
    in_dim = size(W, 2)
    W_2d = reshape(W, out_dim, :)
    num_sorted_blocks = length(sorted_head_ids)

    key = (:bwd_dx, eltype(dY), tensorcore, accumulate,
           nextpow.(4, (in_dim, out_dim, num_sorted_blocks)), TILE_M, mode)

    autotune_launch(indexed_proj_bwd_dx_kernel,
        CartesianSpace(
            TILE_K=(32, 64, 128), TILE_N=(32, 64, 128),
            GROUP_SIZE_M=(8,), occupancy=(1,),
        ),
        cfg -> (num_sorted_blocks * cld(in_dim, cfg.TILE_K),),
        cfg -> (
            W_2d, dY, dX, sorted_ids, sorted_head_ids,
            Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_K), Constant(cfg.TILE_N), Constant(TILE_M),
            Constant(cfg.GROUP_SIZE_M),
            Constant(in_dim),
            Constant(k_slots), Constant(mode),
        );
        key, verify,
    )
    return nothing
end

# ── Backward dW kernel ──────────────────────────────────────────────
#
# dW[:, :, h] = Σ_{entries in head h} dY[:, dy_id] @ X[:, x_id]ᵀ
#
# Grid: (cld(out,TILE_N) * cld(in,TILE_K), H)
# One block per (out_tile, in_tile, head). Iterates over sorted blocks
# for this head via head_block_starts. No atomics needed.
#
# mode: 0 = scatter_proj dW (dY indexed by flat_id, X by tok_col)
#        1 = gather_proj dW  (dY indexed by tok_col, X by flat_id)

function indexed_proj_bwd_dw_kernel(
    X::TileMatrix,                           # (in, x_cols)
    dY::TileMatrix,                          # (out, dy_cols)
    dW::TileArray3,                          # (out, in, H)
    sorted_ids::TileVector{Int32},           # (padded_M,) — flat kL indices
    head_block_starts::TileVector{Int32},    # (H+1,)
    Tc::Type, Tacc::Type,
    TILE_N::Int, TILE_K::Int, TILE_M::Int,
    GROUP_SIZE_M::Int,
    k_slots::Int,
    mode::Int,
)
    padding_mode = ct.PaddingMode.Zero
    bid_nk = ct.bid(1)
    head = ct.bid(2)

    out_dim = size(dW, 1)
    in_dim = size(dW, 2)

    # Decode (bid_n, bid_k) from 1D block index
    bid_n_0, bid_k_0 = swizzle_2d(out_dim, in_dim, TILE_N, TILE_K, GROUP_SIZE_M, bid_nk - 1i32)
    bid_n = bid_n_0 + 1i32
    bid_k = bid_k_0 + 1i32

    # Block range for this head
    blk_start = head_block_starts[head]
    blk_end   = head_block_starts[head + 1i32] - 1i32

    acc = ct.zeros((TILE_N, TILE_K), Tacc)

    # Feature index ranges (fixed for this block)
    n_range = bid_n_0 * Int32(TILE_N) .+ ct.arange((TILE_N,), Int32)
    k_range = bid_k_0 * Int32(TILE_K) .+ ct.arange((TILE_K,), Int32)
    n_valid = n_range .<= out_dim
    k_valid = k_range .<= in_dim
    n_idx = reshape(n_range, (TILE_N, 1))
    k_idx = reshape(k_range, (TILE_K, 1))

    m = Int32(mode)

    blk = blk_start
    while blk <= blk_end
        fids  = ct.load(sorted_ids, (blk,), (TILE_M,); padding_mode)
        valid = fids .>= 1i32
        safe_fids = max.(fids, 1i32)
        tok_cols = (safe_fids .- 1i32) .÷ Int32(k_slots) .+ 1i32

        # Derive dY and X indices based on mode
        inv_m = 1i32 - m
        dy_idx = m .* tok_cols  .+ inv_m .* safe_fids
        x_idx  = m .* safe_fids .+ inv_m .* tok_cols

        # Gather dY — zero OOB rows and padding columns
        dy = ct.gather(dY, (n_idx, reshape(dy_idx, (1, TILE_M))))
        dy = ifelse.(reshape(n_valid, (TILE_N, 1)), dy, Tacc(0))
        dy = ifelse.(reshape(valid, (1, TILE_M)), dy, Tacc(0))

        # Gather X — zero OOB rows
        x = ct.gather(X, (k_idx, reshape(x_idx, (1, TILE_M))))
        x = ifelse.(reshape(k_valid, (TILE_K, 1)), x, Tacc(0))

        # acc += dY @ Xᵀ
        acc = muladd(dy → Tc, (x)ᵀ → Tc, acc)

        blk += 1i32
    end

    ct.store(dW, (bid_n, bid_k, head),
             reshape(acc → eltype(dW), (TILE_N, TILE_K, 1)))
    return
end

function ∇indexed_proj_dw!(
    X, dY, dW,
    sorted_ids, head_block_starts;
    TILE_M, k_slots, mode,
    tensorcore = tensorcore_type(eltype(X)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing,
)
    out_dim, in_dim, H = size(dW)

    key = (:bwd_dw, eltype(X), tensorcore, accumulate,
           nextpow.(4, (out_dim, in_dim)), H, TILE_M, mode)

    autotune_launch(indexed_proj_bwd_dw_kernel,
        CartesianSpace(
            TILE_N=(32, 64, 128), TILE_K=(32, 64, 128),
            GROUP_SIZE_M=(8, 16), occupancy=(1, 2, 4),
        ),
        cfg -> (cld(out_dim, cfg.TILE_N) * cld(in_dim, cfg.TILE_K), H),
        cfg -> (
            X, dY, dW, sorted_ids, head_block_starts,
            Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_N), Constant(cfg.TILE_K), Constant(TILE_M),
            Constant(cfg.GROUP_SIZE_M),
            Constant(k_slots), Constant(mode),
        );
        key, verify,
    )
    return nothing
end

