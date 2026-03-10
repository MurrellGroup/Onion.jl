# Fused SwiGLU + down projection forward and backward kernels.
#
# Forward:  O = W_down @ (silu(W_gate @ X) ⊙ (W_up @ X))
#
# Layout:
#   X:      (D, N)       — hidden × tokens  (N = seq × batch, reshaped by wrapper)
#   W_gate: (K, D)       — intermediate × hidden
#   W_up:   (K, D)       — intermediate × hidden
#   W_down: (D_out, K)   — output × intermediate
#   O:      (D_out, N)

# ── Forward: Step 1 — fused gate + up matmuls (shared X loads) ───────

function swiglu_gate_up_fwd(
    X::TileMatrix, W_gate::TileMatrix, W_up::TileMatrix,
    Gate::TileMatrix, Up::TileMatrix,
    Tc::Type, Tacc::Type,
    TILE_K::Int, TILE_N::Int, TILE_D::Int,
)
    padding_mode = ct.PaddingMode.Zero
    pid = ct.bid(1)

    K = size(W_gate, 1)
    D = size(X, 1)
    N = size(X, 2)

    k_0, n_0 = swizzle_2d(K, N, TILE_K, TILE_N, 8, pid - 1i32)
    k = k_0 + 1i32
    n = n_0 + 1i32
    num_d = cld(D, Int32(TILE_D))

    acc_gate = ct.zeros((TILE_K, TILE_N), Tacc)
    acc_up   = ct.zeros((TILE_K, TILE_N), Tacc)

    d = 1i32
    while d <= num_d
        x  = ct.load(X,      (d, n), (TILE_D, TILE_N); padding_mode)
        wg = ct.load(W_gate, (k, d), (TILE_K, TILE_D); padding_mode)
        wu = ct.load(W_up,   (k, d), (TILE_K, TILE_D); padding_mode)
        acc_gate = muladd(wg → Tc, x → Tc, acc_gate)
        acc_up   = muladd(wu → Tc, x → Tc, acc_up)
        d += 1i32
    end

    ct.store(Gate, (k, n), acc_gate → eltype(Gate))
    ct.store(Up,   (k, n), acc_up   → eltype(Up))
    return
end

# ── Forward: Step 2 — fused activation + down projection ────────────

function swiglu_act_down_fwd(
    Gate::TileMatrix, Up::TileMatrix, W_down::TileMatrix,
    O::TileMatrix,
    T::Type, Tc::Type, Tacc::Type,
    TILE_O::Int, TILE_N::Int, TILE_K::Int,
)
    padding_mode = ct.PaddingMode.Zero
    pid = ct.bid(1)

    D_out = size(W_down, 1)
    K     = size(W_down, 2)
    N     = size(Gate, 2)

    o_0, n_0 = swizzle_2d(D_out, N, TILE_O, TILE_N, 8, pid - 1i32)
    o = o_0 + 1i32
    n = n_0 + 1i32
    num_k = cld(K, Int32(TILE_K))

    acc_out = ct.zeros((TILE_O, TILE_N), Tacc)

    k = 1i32
    while k <= num_k
        gate = ct.load(Gate,   (k, n), (TILE_K, TILE_N); padding_mode) → T
        up   = ct.load(Up,     (k, n), (TILE_K, TILE_N); padding_mode) → T
        wd   = ct.load(W_down, (o, k), (TILE_O, TILE_K); padding_mode)

        sig = 1 ./ (1 .+ exp.(0 .- gate))
        a = gate .* sig .* up

        acc_out = muladd(wd → Tc, a → Tc, acc_out)
        k += 1i32
    end

    ct.store(O, (o, n), acc_out → eltype(O))
    return
end

# ── Backward: dA = W_downᵀ @ Ō  (standalone matmul) ─────────────────

function swiglu_bwd_dA(
    W_down::TileMatrix, Ō::TileMatrix, dA_out::TileMatrix,
    Tc::Type, Tacc::Type,
    TILE_K::Int, TILE_N::Int, TILE_O::Int,
)
    padding_mode = ct.PaddingMode.Zero
    pid = ct.bid(1)

    K = size(dA_out, 1); N = size(dA_out, 2); D_out = size(Ō, 1)

    k_0, n_0 = swizzle_2d(K, N, TILE_K, TILE_N, 32, pid - 1i32)
    k = k_0 + 1i32; n = n_0 + 1i32
    num_o = cld(D_out, Int32(TILE_O))

    dA_acc = ct.zeros((TILE_K, TILE_N), Tacc)

    o = 1i32
    while o <= num_o
        wd = ct.load(W_down, (o, k), (TILE_O, TILE_K); padding_mode)
        ō  = ct.load(Ō,      (o, n), (TILE_O, TILE_N); padding_mode)
        dA_acc = muladd((wd)ᵀ → Tc, ō → Tc, dA_acc)
        o += 1i32
    end

    ct.store(dA_out, (k, n), dA_acc → eltype(dA_out))
    return
end

# ── Backward: dX (fused activation grads from precomputed dA) ───────

function swiglu_bwd_dX(
    W_gate::TileMatrix, W_up::TileMatrix,
    Gate::TileMatrix, Up::TileMatrix,
    dA::TileMatrix, X̄::TileMatrix,
    T::Type, Tc::Type, Tacc::Type,
    TILE_D::Int, TILE_N::Int, TILE_K::Int,
)
    padding_mode = ct.PaddingMode.Zero
    pid = ct.bid(1)

    D = size(X̄, 1); N = size(X̄, 2); K = size(Gate, 1)

    d_0, n_0 = swizzle_2d(D, N, TILE_D, TILE_N, 8, pid - 1i32)
    d = d_0 + 1i32; n = n_0 + 1i32
    num_k = cld(K, Int32(TILE_K))

    x̄_acc = ct.zeros((TILE_D, TILE_N), Tacc)

    k = 1i32
    while k <= num_k
        da   = ct.load(dA,   (k, n), (TILE_K, TILE_N); padding_mode) → T
        gate = ct.load(Gate, (k, n), (TILE_K, TILE_N); padding_mode) → T
        up   = ct.load(Up,   (k, n), (TILE_K, TILE_N); padding_mode) → T

        sig       = 1 ./ (1 .+ exp.(0 .- gate))
        silu_gate = gate .* sig
        dsilu     = sig .* (1 .+ gate .* (1 .- sig))

        d_up   = da .* silu_gate
        d_gate = da .* up .* dsilu

        wg = ct.load(W_gate, (k, d), (TILE_K, TILE_D); padding_mode)
        wu = ct.load(W_up,   (k, d), (TILE_K, TILE_D); padding_mode)

        x̄_acc = muladd((wg)ᵀ → Tc, d_gate → Tc, x̄_acc)
        x̄_acc = muladd((wu)ᵀ → Tc, d_up   → Tc, x̄_acc)

        k += 1i32
    end

    ct.store(X̄, (d, n), x̄_acc → eltype(X̄))
    return
end

# ── Backward: dW_down ────────────────────────────────────────────────

function swiglu_ffn_bwd_dw_down(
    Gate::TileMatrix, Up::TileMatrix,
    Ō::TileMatrix, W̄_down::TileMatrix,
    T::Type, Tc::Type, Tacc::Type,
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

    w̄d_acc = ct.zeros((TILE_O, TILE_K), Tacc)

    n = 1i32
    while n <= num_n
        gate = ct.load(Gate, (k, n), (TILE_K, TILE_N); padding_mode) → T
        up   = ct.load(Up,   (k, n), (TILE_K, TILE_N); padding_mode) → T
        ō    = ct.load(Ō,    (o, n), (TILE_O, TILE_N); padding_mode)

        sig = 1 ./ (1 .+ exp.(0 .- gate))
        a = gate .* sig .* up

        w̄d_acc = muladd(ō → Tc, (a)ᵀ → Tc, w̄d_acc)

        n += 1i32
    end

    ct.store(W̄_down, (o, k), w̄d_acc → eltype(W̄_down))

    return
end

# ── Backward: dW_gate, dW_up (fused activation grads from precomputed dA) ──

function swiglu_bwd_dW(
    X::TileMatrix, Gate::TileMatrix, Up::TileMatrix,
    dA::TileMatrix,
    W̄_gate::TileMatrix, W̄_up::TileMatrix,
    T::Type, Tc::Type, Tacc::Type,
    TILE_K::Int, TILE_D::Int, TILE_N::Int,
)
    padding_mode = ct.PaddingMode.Zero
    pid = ct.bid(1)

    K = size(W̄_gate, 1); D = size(W̄_gate, 2); N = size(X, 2)

    k_0, d_0 = swizzle_2d(K, D, TILE_K, TILE_D, 8, pid - 1i32)
    k = k_0 + 1i32; d = d_0 + 1i32
    num_n = cld(N, Int32(TILE_N))

    w̄g_acc = ct.zeros((TILE_K, TILE_D), Tacc)
    w̄u_acc = ct.zeros((TILE_K, TILE_D), Tacc)

    n = 1i32
    while n <= num_n
        da   = ct.load(dA,   (k, n), (TILE_K, TILE_N); padding_mode) → T
        gate = ct.load(Gate, (k, n), (TILE_K, TILE_N); padding_mode) → T
        up   = ct.load(Up,   (k, n), (TILE_K, TILE_N); padding_mode) → T

        sig       = 1 ./ (1 .+ exp.(0 .- gate))
        silu_gate = gate .* sig
        dsilu     = sig .* (1 .+ gate .* (1 .- sig))

        d_up   = da .* silu_gate
        d_gate = da .* up .* dsilu

        x = ct.load(X, (d, n), (TILE_D, TILE_N); padding_mode)

        w̄g_acc = muladd(d_gate → Tc, (x)ᵀ → Tc, w̄g_acc)
        w̄u_acc = muladd(d_up   → Tc, (x)ᵀ → Tc, w̄u_acc)

        n += 1i32
    end

    ct.store(W̄_gate, (k, d), w̄g_acc → eltype(W̄_gate))
    ct.store(W̄_up,   (k, d), w̄u_acc → eltype(W̄_up))
    return
end

# ── Wrappers ─────────────────────────────────────────────────────────

function swiglu_ffn!(O::AbstractMatrix,
    X::AbstractMatrix,
    W_gate::AbstractMatrix,
    W_up::AbstractMatrix,
    W_down::AbstractMatrix;
    arithmetic = arithmetic_type(eltype(X)),
    tensorcore = tensorcore_type(eltype(X)),
    accumulate = accumulate_type(tensorcore),
    verify = nothing,
)
    D, N = size(X)
    K, D2 = size(W_gate)
    D_out, K2 = size(W_down)
    @assert D == D2
    @assert K == K2
    @assert size(W_gate) == size(W_up)
    @assert size(O) == (D_out, N)

    # Always allocate Gate/Up (needed as intermediates between the two kernels)
    Gate = similar(X, K, N)
    Up   = similar(X, K, N)

    key = (eltype(X), arithmetic, tensorcore, accumulate)

    # Step 1: fused gate + up matmuls (shared X loads)
    autotune_launch(swiglu_gate_up_fwd,
        CartesianSpace(
            TILE_K=(32, 64, 128), TILE_N=(32, 64, 128),
            TILE_D=(32, 64),
            occupancy=(1, 2, 4),
        ),
        cfg -> (cld(K, cfg.TILE_K) * cld(N, cfg.TILE_N),),
        cfg -> (
            X, W_gate, W_up, Gate, Up,
            Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_K), Constant(cfg.TILE_N), Constant(cfg.TILE_D),
        );
        key
    )

    # Step 2: fused activation + down projection
    autotune_launch(swiglu_act_down_fwd,
        CartesianSpace(
            TILE_O=(32, 64, 128), TILE_N=(32, 64, 128),
            TILE_K=(32, 64),
            occupancy=(1, 2, 4),
        ),
        cfg -> (cld(D_out, cfg.TILE_O) * cld(N, cfg.TILE_N),),
        cfg -> (
            Gate, Up, W_down, O,
            Constant(arithmetic), Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_O), Constant(cfg.TILE_N), Constant(cfg.TILE_K),
        );
        key, verify
    )

    return Gate, Up
end



function ∇swiglu_ffn!(
    X̄::AbstractMatrix,
    W̄_gate::AbstractMatrix, W̄_up::AbstractMatrix, W̄_down::AbstractMatrix,
    Ō::AbstractMatrix,
    X::AbstractMatrix,
    W_gate::AbstractMatrix, W_up::AbstractMatrix, W_down::AbstractMatrix,
    Gate::AbstractMatrix, Up::AbstractMatrix;
    arithmetic = arithmetic_type(eltype(X)),
    tensorcore = tensorcore_type(eltype(X)),
    accumulate = accumulate_type(tensorcore),
    verify_x = nothing,
    verify_w_down = nothing,
    verify_w = nothing
)
    D, N = size(X)
    K, _ = size(W_gate)
    D_out, _ = size(W_down)

    key = (eltype(X), arithmetic, tensorcore, accumulate)

    # Allocate dA intermediate: dA = W_downᵀ @ Ō, shape (K, N)
    dA = similar(Gate)

    # 1. dA = W_downᵀ @ Ō
    autotune_launch(swiglu_bwd_dA,
        CartesianSpace(
            TILE_K=(32, 64, 128, 256), TILE_N=(32, 64, 128, 256),
            TILE_O=(32, 64, 128),
            occupancy=(1, 2, 4),
        ),
        cfg -> (cld(K, cfg.TILE_K) * cld(N, cfg.TILE_N),),
        cfg -> (
            W_down, Ō, dA,
            Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_K), Constant(cfg.TILE_N), Constant(cfg.TILE_O),
        );
        key
    )

    # 2. dX = W_gateᵀ @ d_gate + W_upᵀ @ d_up  (fused activation grads from dA)
    autotune_launch(swiglu_bwd_dX,
        CartesianSpace(
            TILE_D=(32, 64, 128, 256), TILE_N=(32, 64, 128, 256),
            TILE_K=(32, 64, 128),
            occupancy=(1, 2, 4),
        ),
        cfg -> (cld(D, cfg.TILE_D) * cld(N, cfg.TILE_N),),
        cfg -> (
            W_gate, W_up, Gate, Up, dA, X̄,
            Constant(arithmetic), Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_D), Constant(cfg.TILE_N), Constant(cfg.TILE_K),
        );
        key, verify=verify_x
    )

    # 3. dW_down = Ō @ Aᵀ  (fused silu — unchanged)
    autotune_launch(swiglu_ffn_bwd_dw_down,
        CartesianSpace(
            TILE_O=(32, 64, 128), TILE_K=(32, 64, 128, 256),
            TILE_N=(32, 64, 128),
            occupancy=(1, 2, 4),
        ),
        cfg -> (cld(D_out, cfg.TILE_O) * cld(K, cfg.TILE_K),),
        cfg -> (
            Gate, Up, Ō, W̄_down,
            Constant(arithmetic), Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_O), Constant(cfg.TILE_K), Constant(cfg.TILE_N),
        );
        key, verify=verify_w_down
    )

    # 4. dW_gate, dW_up  (fused activation grads from dA, shared X loads)
    autotune_launch(swiglu_bwd_dW,
        CartesianSpace(
            TILE_K=(32, 64, 128, 256), TILE_D=(32, 64, 128, 256),
            TILE_N=(32, 64, 128),
            occupancy=(1, 2, 4),
        ),
        cfg -> (cld(K, cfg.TILE_K) * cld(D, cfg.TILE_D),),
        cfg -> (
            X, Gate, Up, dA, W̄_gate, W̄_up,
            Constant(arithmetic), Constant(tensorcore), Constant(accumulate),
            Constant(cfg.TILE_K), Constant(cfg.TILE_D), Constant(cfg.TILE_N),
        );
        key, verify=verify_w
    )

    return X̄, W̄_gate, W̄_up, W̄_down
end
