# K, U: (I, D, H) — gate/up weights in (d_inter, d_head, n_heads) layout
# V:    (D, I, H) — down weight in (d_head, d_inter, n_heads) layout
# Q, O: (D, k*L) — flattened input/output
# R:    (E, k*L) — optional expert routing weights (nothing for non-expert mode)
# sorted_ids:      (padded_M,) — flat indices per element (0 = padding)
# sorted_head_ids: (num_blocks,) — head id per TILE_M block

function top_mhffn_fwd(
    Q::TileMatrix, K::TileArray3, U::TileArray3, V::TileArray3,
    O::TileMatrix,
    R::Optional{TileMatrix},
    sorted_ids::TileVector{Int32},
    sorted_head_ids::TileVector{Int32},
    Tc::Type, Tacc::Type,
    TILE_D::Int, TILE_M::Int, TILE_I::Int,
    D_E::Optional{Int},
)
    padding_mode = ct.PaddingMode.Zero
    bid = ct.bid(1)

    I_dim = size(K, 1)
    head = sorted_head_ids[bid]

    fids      = ct.load(sorted_ids, (bid,), (TILE_M,); padding_mode)
    valid     = fids .>= 1i32
    safe_fids = max.(fids, 1i32)

    d_range = ct.arange((TILE_D,), Int32)
    q = ct.gather(Q, (reshape(d_range, (TILE_D, 1)),
                       reshape(safe_fids, (1, TILE_M))))

    acc = ct.zeros((TILE_D, TILE_M), Tacc)
    num_i = cld(I_dim, Int32(TILE_I))

    if R isa TileArray
        tiles_per_expert = D_E ÷ TILE_I
    end

    i = 1i32
    while i <= num_i
        k_tile = ct.load(K, (i, 1, head), (TILE_I, TILE_D); padding_mode)
        u_tile = ct.load(U, (i, 1, head), (TILE_I, TILE_D); padding_mode)
        v_tile = ct.load(V, (1, i, head), (TILE_D, TILE_I); padding_mode)

        m = muladd(k_tile → Tc, q → Tc, ct.zeros((TILE_I, TILE_M), Tacc))
        n = muladd(u_tile → Tc, q → Tc, ct.zeros((TILE_I, TILE_M), Tacc))

        a = m ./ (1 .+ exp.(0 .- m)) .* n

        if R isa TileArray
            e = fld(i - 1i32, tiles_per_expert) + 1i32
            r = ct.gather(R, (reshape(ct.broadcast_to(ct.Tile(e), (1,)), (1, 1)),
                               reshape(safe_fids, (1, TILE_M))))
            r = ifelse.(reshape(valid, (1, TILE_M)), r, Tacc(0))
            a = a .* r
        end

        acc = muladd(v_tile → Tc, a → Tc, acc)

        i += 1i32
    end

    acc = ifelse.(reshape(valid, (1, TILE_M)), acc, Tacc(0))
    scatter_ids = ifelse.(valid, safe_fids, Int32(0))
    ct.scatter(O, (reshape(d_range, (TILE_D, 1)),
                    reshape(scatter_ids, (1, TILE_M))),
               acc → eltype(O))
    return
end

# ── Backward dQ kernel ────────────────────────────────────────────────
# Grid: 1D (num_sorted_blocks,) — same as forward.
# Each block gathers Q and dO, recomputes activations, accumulates dQ.
# Also computes dR when R is provided.

function top_mhffn_bwd_dq(
    Q::TileMatrix, K::TileArray3, U::TileArray3, V::TileArray3,
    Ō::TileMatrix, Q̄::TileMatrix,
    R::Optional{TileMatrix}, R̄::Optional{TileMatrix},
    sorted_ids::TileVector{Int32},
    sorted_head_ids::TileVector{Int32},
    Tc::Type, Tacc::Type,
    TILE_D::Int, TILE_M::Int, TILE_I::Int,
    num_i::Int,
    D_E::Optional{Int},
)
    padding_mode = ct.PaddingMode.Zero
    bid = ct.bid(1)

    head = sorted_head_ids[bid]

    fids      = ct.load(sorted_ids, (bid,), (TILE_M,); padding_mode)
    valid     = fids .>= 1i32
    safe_fids = max.(fids, 1i32)

    d_range = ct.arange((TILE_D,), Int32)
    q = ct.gather(Q, (reshape(d_range, (TILE_D, 1)),
                       reshape(safe_fids, (1, TILE_M))))
    ō = ct.gather(Ō, (reshape(d_range, (TILE_D, 1)),
                       reshape(safe_fids, (1, TILE_M))))

    q̄_acc = ct.zeros((TILE_D, TILE_M), Tacc)

    if R isa TileArray
        tiles_per_expert = D_E ÷ TILE_I
        dr_acc = ct.zeros((1, TILE_M), Tacc)
    end

    i = 1i32
    while i <= num_i
        k_tile = ct.load(K, (i, 1, head), (TILE_I, TILE_D); padding_mode)
        u_tile = ct.load(U, (i, 1, head), (TILE_I, TILE_D); padding_mode)
        v_tile = ct.load(V, (1, i, head), (TILE_D, TILE_I); padding_mode)

        m = muladd(k_tile → Tc, q → Tc, ct.zeros((TILE_I, TILE_M), Tacc))
        n = muladd(u_tile → Tc, q → Tc, ct.zeros((TILE_I, TILE_M), Tacc))

        sig = 1 ./ (1 .+ exp.(0 .- m))
        silu_m = m .* sig

        ā = muladd((v_tile)ᵀ → Tc, ō → Tc, ct.zeros((TILE_I, TILE_M), Tacc))

        dsilu_dm = sig .* (1 .+ m .* (1 .- sig))

        if R isa TileArray
            e = fld(i - 1i32, tiles_per_expert) + 1i32
            r = ct.gather(R, (reshape(ct.broadcast_to(ct.Tile(e), (1,)), (1, 1)),
                               reshape(safe_fids, (1, TILE_M))))
            r = ifelse.(reshape(valid, (1, TILE_M)), r, Tacc(0))

            dr_acc = dr_acc .+ sum(ā .* silu_m .* n, dims=1)

            if iszero(mod(i, tiles_per_expert))
                dr_acc = ifelse.(reshape(valid, (1, TILE_M)), dr_acc, Tacc(0))
                scatter_ids = ifelse.(valid, safe_fids, Int32(0))
                ct.scatter(R̄, (reshape(ct.broadcast_to(ct.Tile(e), (1,)), (1, 1)),
                                reshape(scatter_ids, (1, TILE_M))),
                           dr_acc → eltype(R̄))
                dr_acc = ct.zeros((1, TILE_M), Tacc)
            end

            M̄ = ā .* n .* dsilu_dm .* r
            N̄ = ā .* silu_m .* r
        else
            M̄ = ā .* n .* dsilu_dm
            N̄ = ā .* silu_m
        end

        q̄_acc = muladd((k_tile)ᵀ → Tc, M̄ → Tc, q̄_acc)
        q̄_acc = muladd((u_tile)ᵀ → Tc, N̄ → Tc, q̄_acc)

        i += 1i32
    end

    q̄_acc = ifelse.(reshape(valid, (1, TILE_M)), q̄_acc, Tacc(0))
    scatter_ids = ifelse.(valid, safe_fids, Int32(0))
    ct.scatter(Q̄, (reshape(d_range, (TILE_D, 1)),
                    reshape(scatter_ids, (1, TILE_M))),
               q̄_acc → eltype(Q̄))
    return
end

# ── Backward dKUV kernel ─────────────────────────────────────────────
# Grid: 2D (cld(I, TILE_I), H)
# Each block accumulates dK, dU, dV for one (I-tile, head) by iterating
# over all dispatch blocks belonging to that head via head_block_starts.

function top_mhffn_bwd_dkuv(
    Q::TileMatrix, K::TileArray3, U::TileArray3, V::TileArray3,
    Ō::TileMatrix,
    R::Optional{TileMatrix},
    K̄::TileArray3, Ū::TileArray3, V̄::TileArray3,
    sorted_ids::TileVector{Int32},
    sorted_head_ids::TileVector{Int32},
    head_block_starts::TileVector{Int32},
    Tc::Type, Tacc::Type,
    TILE_D::Int, TILE_M::Int, TILE_I::Int,
    D_E::Optional{Int},
)
    padding_mode = ct.PaddingMode.Zero
    i_bid = ct.bid(1)
    head  = ct.bid(2)

    k_tile = ct.load(K, (i_bid, 1i32, head), (TILE_I, TILE_D, 1); padding_mode)
    k_tile = dropdims(k_tile, dims=3)
    u_tile = ct.load(U, (i_bid, 1i32, head), (TILE_I, TILE_D, 1); padding_mode)
    u_tile = dropdims(u_tile, dims=3)
    v_tile = ct.load(V, (1i32, i_bid, head), (TILE_D, TILE_I, 1); padding_mode)
    v_tile = dropdims(v_tile, dims=3)

    k̄_acc = ct.zeros((TILE_I, TILE_D), Tacc)
    ū_acc = ct.zeros((TILE_I, TILE_D), Tacc)
    v̄_acc = ct.zeros((TILE_D, TILE_I), Tacc)

    if R isa TileArray
        tiles_per_expert = D_E ÷ TILE_I
        e = fld(i_bid - 1i32, tiles_per_expert) + 1i32
    end

    blk_start = head_block_starts[head]
    blk_end   = head_block_starts[head + 1i32] - 1i32

    d_range = ct.arange((TILE_D,), Int32)

    blk = blk_start
    while blk <= blk_end
        fids      = ct.load(sorted_ids, (blk,), (TILE_M,); padding_mode)
        valid     = fids .>= 1i32
        safe_fids = max.(fids, 1i32)

        q = ct.gather(Q, (reshape(d_range, (TILE_D, 1)),
                           reshape(safe_fids, (1, TILE_M))))
        q = ifelse.(reshape(valid, (1, TILE_M)), q, Tacc(0))
        ō = ct.gather(Ō, (reshape(d_range, (TILE_D, 1)),
                           reshape(safe_fids, (1, TILE_M))))
        ō = ifelse.(reshape(valid, (1, TILE_M)), ō, Tacc(0))

        m = muladd(k_tile → Tc, q → Tc, ct.zeros((TILE_I, TILE_M), Tacc))
        n = muladd(u_tile → Tc, q → Tc, ct.zeros((TILE_I, TILE_M), Tacc))

        sig = 1 ./ (1 .+ exp.(0 .- m))
        silu_m = m .* sig
        a = silu_m .* n

        ā = muladd((v_tile)ᵀ → Tc, ō → Tc, ct.zeros((TILE_I, TILE_M), Tacc))

        dsilu_dm = sig .* (1 .+ m .* (1 .- sig))

        if R isa TileArray
            r = ct.gather(R, (reshape(ct.broadcast_to(ct.Tile(e), (1,)), (1, 1)),
                               reshape(safe_fids, (1, TILE_M))))
            r = ifelse.(reshape(valid, (1, TILE_M)), r, Tacc(0))
            a = a .* r
            M̄ = ā .* n .* dsilu_dm .* r
            N̄ = ā .* silu_m .* r
        else
            M̄ = ā .* n .* dsilu_dm
            N̄ = ā .* silu_m
        end

        k̄_acc = muladd(M̄ → Tc, (q)ᵀ → Tc, k̄_acc)
        ū_acc = muladd(N̄ → Tc, (q)ᵀ → Tc, ū_acc)
        v̄_acc = muladd(ō → Tc, (a)ᵀ → Tc, v̄_acc)

        blk += 1i32
    end

    ct.store(K̄, (i_bid, 1i32, head), reshape(k̄_acc, (TILE_I, TILE_D, 1)) → eltype(K̄))
    ct.store(Ū, (i_bid, 1i32, head), reshape(ū_acc, (TILE_I, TILE_D, 1)) → eltype(Ū))
    ct.store(V̄, (1i32, i_bid, head), reshape(v̄_acc, (TILE_D, TILE_I, 1)) → eltype(V̄))

    return
end

# ── Launchers ────────────────────────────────────────────────────────

function top_mhffn_fwd!(O,
    Q, K, U, V,
    sorted_ids, sorted_head_ids;
    R = nothing,
    D_E = nothing,
    TILE_M,
    tensorcore = tensorcore_type(eltype(Q)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing,
)
    D = size(Q, 1)
    I = size(K, 1)
    num_sorted_blocks = length(sorted_head_ids)

    if !isnothing(R)
        @assert !isnothing(D_E) "D_E required when R is provided"
        tile_i_opts = Tuple(filter(t -> iszero(D_E % t), (32, 64, 128)))
        @assert !isempty(tile_i_opts) "D_E=$D_E must be divisible by at least one of (32, 64, 128)"
    else
        tile_i_opts = (32, 64, 128)
    end

    key = (eltype(Q), tensorcore, accumulate, D, nextpow(4, num_sorted_blocks), !isnothing(R))

    autotune_launch(top_mhffn_fwd,
        CartesianSpace(TILE_I=tile_i_opts, occupancy=(1, 2, 4)),
        cfg -> (num_sorted_blocks,),
        cfg -> (
            Q, K, U, V, O, R, sorted_ids, sorted_head_ids,
            Constant(tensorcore), Constant(accumulate),
            Constant(D), Constant(TILE_M), Constant(cfg.TILE_I),
            D_E,
        );
        key, verify,
    )

    return nothing
end

function top_mhffn_bwd_dq!(Q̄,
    Q, K, U, V, Ō,
    sorted_ids, sorted_head_ids;
    R = nothing, R̄ = nothing,
    D_E = nothing,
    TILE_M,
    tensorcore = tensorcore_type(eltype(Q)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing,
)
    D = size(Q, 1)
    I = size(K, 1)
    num_sorted_blocks = length(sorted_head_ids)

    if !isnothing(R)
        @assert !isnothing(R̄) "R̄ required when R is provided"
        @assert !isnothing(D_E) "D_E required when R is provided"
        tile_i_opts = Tuple(filter(t -> iszero(D_E % t), (32, 64, 128)))
        @assert !isempty(tile_i_opts)
    else
        tile_i_opts = (32, 64, 128)
    end

    key = (:bwd_dq, eltype(Q), tensorcore, accumulate, D, nextpow(4, num_sorted_blocks), !isnothing(R))

    autotune_launch(top_mhffn_bwd_dq,
        CartesianSpace(TILE_I=tile_i_opts, occupancy=(1, 2, 4)),
        cfg -> (num_sorted_blocks,),
        cfg -> (
            Q, K, U, V, Ō, Q̄,
            R, R̄,
            sorted_ids, sorted_head_ids,
            Constant(tensorcore), Constant(accumulate),
            Constant(D), Constant(TILE_M), Constant(cfg.TILE_I),
            Constant(cld(I, cfg.TILE_I)),
            D_E,
        );
        key, verify,
    )

    return nothing
end

function top_mhffn_bwd_dkuv!(K̄, Ū, V̄,
    Q, K, U, V, Ō,
    sorted_ids, sorted_head_ids, head_block_starts;
    R = nothing,
    D_E = nothing,
    TILE_M,
    tensorcore = tensorcore_type(eltype(Q)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing,
)
    D = size(Q, 1)
    I = size(K, 1)
    H = size(K, 3)

    if !isnothing(R)
        @assert !isnothing(D_E)
        tile_i_opts = Tuple(filter(t -> iszero(D_E % t), (32, 64, 128)))
        @assert !isempty(tile_i_opts)
    else
        tile_i_opts = (32, 64, 128)
    end

    key = (:bwd_dkuv, eltype(Q), tensorcore, accumulate, D, H, !isnothing(R))

    autotune_launch(top_mhffn_bwd_dkuv,
        CartesianSpace(TILE_I=tile_i_opts, occupancy=(1, 2, 4)),
        cfg -> (cld(I, cfg.TILE_I), H),
        cfg -> (
            Q, K, U, V, Ō, R, K̄, Ū, V̄,
            sorted_ids, sorted_head_ids, head_block_starts,
            Constant(tensorcore), Constant(accumulate),
            Constant(D), Constant(TILE_M), Constant(cfg.TILE_I),
            D_E,
        );
        key, verify,
    )

    return nothing
end
