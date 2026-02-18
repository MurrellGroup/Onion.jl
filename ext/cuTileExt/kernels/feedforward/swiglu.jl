# Fused SwiGLU + down projection forward and backward kernels.
#
# Forward:  O = W_down @ (silu(W_gate @ X) ⊙ (W_up @ X))
#
# Layout (Flux Dense convention):
#   X:      (D, N)       — hidden × tokens  (N = seq × batch, reshaped by wrapper)
#   W_gate: (K, D)       — intermediate × hidden
#   W_up:   (K, D)       — intermediate × hidden
#   W_down: (D_out, K)   — output × intermediate
#   O:      (D_out, N)
#
# see https://github.com/fattorib/fusedswiglu

@inline function swizzle_2d(M, N, tm, tn, GROUP_SIZE_M, bid)
    num_bid_m = cld(M, Int32(tm))
    num_bid_n = cld(N, Int32(tn))
    num_bid_in_group = Int32(GROUP_SIZE_M) * num_bid_n
    group_id = fld(bid, num_bid_in_group)
    first_bid_m = group_id * Int32(GROUP_SIZE_M)
    group_size_m = min(num_bid_m - first_bid_m, Int32(GROUP_SIZE_M))
    bid_m = first_bid_m + rem(bid, group_size_m)
    bid_n = fld(rem(bid, num_bid_in_group), group_size_m)
    return bid_m, bid_n
end

# ── Forward ──────────────────────────────────────────────────────────

function swiglu_ffn_fwd_kernel(
    X::TileMatrix, W_gate::TileMatrix, W_up::TileMatrix, W_down::TileMatrix,
    O::TileMatrix, Gate::TileMatrix, Up::TileMatrix,
    T::Type,
    TILE_O::Int, TILE_N::Int, TILE_K::Int, TILE_D::Int,
    SAVE::Bool,
)
    padding_mode = ct.PaddingMode.Zero
    pid = ct.bid(1)

    D_out = size(W_down, 1)
    D     = size(X, 1)
    N     = size(X, 2)
    K     = size(W_gate, 1)

    o_0, n_0 = swizzle_2d(D_out, N, TILE_O, TILE_N, 8, pid - 1i32)
    o = o_0 + 1i32
    n = n_0 + 1i32

    num_k = cld(K, Int32(TILE_K))
    num_d = cld(D, Int32(TILE_D))

    acc_out = ct.zeros((TILE_O, TILE_N), Float32)

    k = 1i32
    while k <= num_k
        acc_gate = ct.zeros((TILE_K, TILE_N), Float32)
        acc_up   = ct.zeros((TILE_K, TILE_N), Float32)

        d = 1i32
        while d <= num_d
            x  = ct.load(X,      (d, n), (TILE_D, TILE_N); padding_mode)
            wg = ct.load(W_gate, (k, d), (TILE_K, TILE_D); padding_mode)
            wu = ct.load(W_up,   (k, d), (TILE_K, TILE_D); padding_mode)

            acc_gate = muladd(wg → T, x → T, acc_gate)
            acc_up   = muladd(wu → T, x → T, acc_up)

            d += 1i32
        end

        if SAVE
            if o == 1i32
                ct.store(Gate, (k, n), acc_gate → eltype(Gate))
                ct.store(Up,   (k, n), acc_up   → eltype(Up))
            end
        end

        sig = 1 ./ (1 .+ exp.(0 .- acc_gate))
        a = acc_gate .* sig .* acc_up

        wd = ct.load(W_down, (o, k), (TILE_O, TILE_K); padding_mode)
        acc_out = muladd(wd → T, a → T, acc_out)

        k += 1i32
    end

    ct.store(O, (o, n), acc_out → eltype(O))

    return
end

# ── Backward: dX ─────────────────────────────────────────────────────

function swiglu_ffn_bwd_dx_kernel(
    W_gate::TileMatrix, W_up::TileMatrix, W_down::TileMatrix,
    Gate::TileMatrix, Up::TileMatrix,
    Ō::TileMatrix, X̄::TileMatrix,
    T::Type,
    TILE_D::Int, TILE_N::Int, TILE_K::Int, TILE_O::Int,
)
    padding_mode = ct.PaddingMode.Zero
    pid = ct.bid(1)

    D     = size(X̄, 1)
    N     = size(X̄, 2)
    K     = size(Gate, 1)
    D_out = size(Ō, 1)

    d_0, n_0 = swizzle_2d(D, N, TILE_D, TILE_N, 8, pid - 1i32)
    d = d_0 + 1i32
    n = n_0 + 1i32

    num_k = cld(K, Int32(TILE_K))
    num_o = cld(D_out, Int32(TILE_O))

    x̄_acc = ct.zeros((TILE_D, TILE_N), Float32)

    k = 1i32
    while k <= num_k
        dA = ct.zeros((TILE_K, TILE_N), Float32)
        o = 1i32
        while o <= num_o
            wd = ct.load(W_down, (o, k), (TILE_O, TILE_K); padding_mode)
            ō  = ct.load(Ō, (o, n),     (TILE_O, TILE_N); padding_mode)
            dA = muladd((wd)ᵀ → T, ō → T, dA)
            o += 1i32
        end

        gate = ct.load(Gate, (k, n), (TILE_K, TILE_N); padding_mode) → Float32
        up   = ct.load(Up,   (k, n), (TILE_K, TILE_N); padding_mode) → Float32

        sig       = 1 ./ (1 .+ exp.(0 .- gate))
        silu_gate = gate .* sig
        dsilu     = sig .* (1 .+ gate .* (1 .- sig))

        d_up   = dA .* silu_gate
        d_gate = dA .* up .* dsilu

        wg = ct.load(W_gate, (k, d), (TILE_K, TILE_D); padding_mode)
        wu = ct.load(W_up,   (k, d), (TILE_K, TILE_D); padding_mode)

        x̄_acc = muladd((wg)ᵀ → T, d_gate → T, x̄_acc)
        x̄_acc = muladd((wu)ᵀ → T, d_up   → T, x̄_acc)

        k += 1i32
    end

    ct.store(X̄, (d, n), x̄_acc → eltype(X̄))

    return
end

# ── Backward: dW_down ────────────────────────────────────────────────

function swiglu_ffn_bwd_dw_down_kernel(
    Gate::TileMatrix, Up::TileMatrix,
    Ō::TileMatrix, W̄_down::TileMatrix,
    T::Type,
    TILE_O::Int, TILE_K::Int, TILE_N::Int,
)
    padding_mode = ct.PaddingMode.Zero
    pid = ct.bid(1)

    D_out = size(W̄_down, 1)
    K     = size(W̄_down, 2)
    N     = size(Ō, 2)

    o_0, k_0 = swizzle_2d(D_out, K, TILE_O, TILE_K, 8, pid - 1i32)
    o = o_0 + 1i32
    k = k_0 + 1i32

    num_n = cld(N, Int32(TILE_N))

    w̄d_acc = ct.zeros((TILE_O, TILE_K), Float32)

    n = 1i32
    while n <= num_n
        gate = ct.load(Gate, (k, n), (TILE_K, TILE_N); padding_mode) → Float32
        up   = ct.load(Up,   (k, n), (TILE_K, TILE_N); padding_mode) → Float32
        ō    = ct.load(Ō,    (o, n), (TILE_O, TILE_N); padding_mode)

        sig = 1 ./ (1 .+ exp.(0 .- gate))
        a = gate .* sig .* up

        w̄d_acc = muladd(ō → T, (a)ᵀ → T, w̄d_acc)

        n += 1i32
    end

    ct.atomic_add(W̄_down, (o, k), w̄d_acc → eltype(W̄_down))

    return
end

# ── Backward: dW_gate, dW_up ─────────────────────────────────────────

function swiglu_ffn_bwd_dw_kernel(
    X::TileMatrix, W_down::TileMatrix,
    Gate::TileMatrix, Up::TileMatrix,
    Ō::TileMatrix,
    W̄_gate::TileMatrix, W̄_up::TileMatrix,
    T::Type,
    TILE_K::Int, TILE_D::Int, TILE_N::Int, TILE_O::Int,
)
    padding_mode = ct.PaddingMode.Zero
    pid = ct.bid(1)

    K     = size(W̄_gate, 1)
    D     = size(W̄_gate, 2)
    N     = size(X, 2)
    D_out = size(Ō, 1)

    k_0, d_0 = swizzle_2d(K, D, TILE_K, TILE_D, 8, pid - 1i32)
    k = k_0 + 1i32
    d = d_0 + 1i32

    num_n = cld(N, Int32(TILE_N))
    num_o = cld(D_out, Int32(TILE_O))

    w̄g_acc = ct.zeros((TILE_K, TILE_D), Float32)
    w̄u_acc = ct.zeros((TILE_K, TILE_D), Float32)

    n = 1i32
    while n <= num_n
        dA = ct.zeros((TILE_K, TILE_N), Float32)
        o = 1i32
        while o <= num_o
            wd = ct.load(W_down, (o, k), (TILE_O, TILE_K); padding_mode)
            ō  = ct.load(Ō, (o, n),     (TILE_O, TILE_N); padding_mode)
            dA = muladd((wd)ᵀ → T, ō → T, dA)
            o += 1i32
        end

        gate = ct.load(Gate, (k, n), (TILE_K, TILE_N); padding_mode) → Float32
        up   = ct.load(Up,   (k, n), (TILE_K, TILE_N); padding_mode) → Float32

        sig       = 1 ./ (1 .+ exp.(0 .- gate))
        silu_gate = gate .* sig
        dsilu     = sig .* (1 .+ gate .* (1 .- sig))

        d_up   = dA .* silu_gate
        d_gate = dA .* up .* dsilu

        x = ct.load(X, (d, n), (TILE_D, TILE_N); padding_mode)

        w̄g_acc = muladd(d_gate → T, (x)ᵀ → T, w̄g_acc)
        w̄u_acc = muladd(d_up   → T, (x)ᵀ → T, w̄u_acc)

        n += 1i32
    end

    ct.atomic_add(W̄_gate, (k, d), w̄g_acc → eltype(W̄_gate))
    ct.atomic_add(W̄_up,   (k, d), w̄u_acc → eltype(W̄_up))

    return
end

# ── Wrappers ─────────────────────────────────────────────────────────

function swiglu_ffn(X, W_gate, W_up, W_down; save = true, compute = eltype(X), verify = nothing)
    dims = size(X)
    D = dims[1]
    N = prod(dims) ÷ D
    K, D2 = size(W_gate)
    D_out, K2 = size(W_down)
    @assert D == D2
    @assert K == K2
    @assert size(W_gate) == size(W_up)

    X_flat = reshape(X, D, N)
    O_flat = similar(X_flat, D_out, N)
    if save
        Gate = similar(X_flat, K, N)
        Up   = similar(X_flat, K, N)
    else
        Gate = O_flat
        Up   = O_flat
    end

    key = (eltype(X), compute)

    autotune_launch(swiglu_fwd_kernel,
        CartesianSpace(
            TILE_O=(64, 128), TILE_N=(64, 128),
            TILE_K=(32, 64), TILE_D=(32, 64),
            occupancy=(1, 2, 4),
        ),
        cfg -> (cld(D_out, cfg.TILE_O) * cld(N, cfg.TILE_N),),
        cfg -> (
            X_flat, W_gate, W_up, W_down, O_flat, Gate, Up,
            Constant(compute),
            Constant(cfg.TILE_O), Constant(cfg.TILE_N), Constant(cfg.TILE_K), Constant(cfg.TILE_D),
            Constant(save),
        );
        key, verify
    )

    O = reshape(O_flat, D_out, Base.tail(dims)...)
    return save ? (O, Gate, Up) : (O, nothing, nothing)
end

function ∇swiglu_ffn(X, W_gate, W_up, W_down, Gate, Up, Ō; compute = eltype(X), verify = nothing)
    dims = size(X)
    D = dims[1]
    N = prod(dims) ÷ D
    K, _ = size(W_gate)
    D_out, _ = size(W_down)

    X_flat = reshape(X, D, N)
    Ō_flat = reshape(Ō, D_out, N)

    X̄_flat = similar(X_flat)
    W̄_gate = similar(W_gate); fill!(W̄_gate, 0)
    W̄_up   = similar(W_up);   fill!(W̄_up, 0)
    W̄_down = similar(W_down); fill!(W̄_down, 0)

    key = (eltype(X), compute)

    autotune_launch(swiglu_bwd_dx_kernel,
        CartesianSpace(
            TILE_D=(32, 64), TILE_N=(64, 128),
            TILE_K=(32, 64), TILE_O=(64, 128),
            occupancy=(1, 2, 4),
        ),
        cfg -> (cld(D, cfg.TILE_D) * cld(N, cfg.TILE_N),),
        cfg -> (
            W_gate, W_up, W_down, Gate, Up, Ō_flat, X̄_flat,
            Constant(compute),
            Constant(cfg.TILE_D), Constant(cfg.TILE_N), Constant(cfg.TILE_K), Constant(cfg.TILE_O),
        );
        key, verify
    )

    autotune_launch(swiglu_bwd_dw_down_kernel,
        CartesianSpace(
            TILE_O=(64, 128), TILE_K=(32, 64),
            TILE_N=(64, 128),
            occupancy=(1, 2, 4),
        ),
        cfg -> (cld(D_out, cfg.TILE_O) * cld(K, cfg.TILE_K),),
        cfg -> (
            Gate, Up, Ō_flat, W̄_down,
            Constant(compute),
            Constant(cfg.TILE_O), Constant(cfg.TILE_K), Constant(cfg.TILE_N),
        );
        key, verify
    )

    autotune_launch(swiglu_bwd_dw_kernel,
        CartesianSpace(
            TILE_K=(32, 64), TILE_D=(32, 64),
            TILE_N=(64, 128), TILE_O=(64, 128),
            occupancy=(1, 2, 4),
        ),
        cfg -> (cld(K, cfg.TILE_K) * cld(D, cfg.TILE_D),),
        cfg -> (
            X_flat, W_down, Gate, Up, Ō_flat, W̄_gate, W̄_up,
            Constant(compute),
            Constant(cfg.TILE_K), Constant(cfg.TILE_D), Constant(cfg.TILE_N), Constant(cfg.TILE_O),
        );
        key, verify
    )

    X̄ = reshape(X̄_flat, dims)
    return X̄, W̄_gate, W̄_up, W̄_down
end
