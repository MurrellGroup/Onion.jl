# Fused DeltaNet decode: L2 norm Q/K + gate/beta + recurrence + RMSNorm + z-gate.
# Grid: (H, B) — one block per head per batch.

function fused_deltanet_decode_fwd(
    Q_raw::TileArray3, K_raw::TileArray3, V::TileArray3,
    Alpha::TileMatrix, BetaRaw::TileMatrix, Z::TileArray3,
    A_log::TileVector, DtBias::TileVector, NormW::TileVector,
    S::TileArray4, O::TileArray3,
    Dk::Int, Dv::Int, NormEps::Float32, BLOCK_DK::Int, V_PER_K::Int, OUT_T::Type,
)
    padding_mode = ct.PaddingMode.Zero
    h_v, b = ct.bid(1), ct.bid(2)  # h_v = value head index
    h_k = cld(h_v, Int32(V_PER_K))  # key head index (GQA)
    num = cld(Int32(Dk), Int32(BLOCK_DK))

    # Gate/beta scalars (indexed by value head)
    alpha_val = Alpha[h_v, b] → Float32
    beta_raw_val = BetaRaw[h_v, b] → Float32
    a_log_val = A_log[h_v] → Float32
    dt_bias_val = DtBias[h_v] → Float32
    sp_input = alpha_val + dt_bias_val
    sp = sp_input > 20.0f0 ? sp_input : log(1.0f0 + exp(sp_input))
    decay = exp(-exp(a_log_val) * sp)
    beta = 1.0f0 / (1.0f0 + exp(-beta_raw_val))

    # L2 norms (Q, K indexed by key head)
    q_ss = ct.zeros((BLOCK_DK,), Float32)
    k_ss = ct.zeros((BLOCK_DK,), Float32)
    i = 1i32
    while i <= num
        qt = ct.load(Q_raw, (i, h_k, b), (BLOCK_DK,); padding_mode) → Float32
        kt = ct.load(K_raw, (i, h_k, b), (BLOCK_DK,); padding_mode) → Float32
        q_ss = q_ss .+ qt .* qt
        k_ss = k_ss .+ kt .* kt
        i += 1i32
    end
    q_scale = 1.0f0 ./ sqrt.(sum(q_ss; dims=1) .+ 1.0f-6) ./ sqrt(Float32(Dk))
    k_scale = 1.0f0 ./ sqrt.(sum(k_ss; dims=1) .+ 1.0f-6)

    v = ct.load(V, (1, h_v, b), (Dv,)) → Float32

    # Pass 1: decay + S^T @ k_norm
    acc = ct.zeros((Dv,), Float32)
    i = 1i32
    while i <= num
        s = ct.load(S, (i, 1, h_v, b), (BLOCK_DK, Dv); padding_mode) → Float32
        kt = ct.load(K_raw, (i, h_k, b), (BLOCK_DK,); padding_mode) → Float32
        s = s .* decay
        ct.store(S, (i, 1, h_v, b), s → eltype(S))
        acc = mva((s)ᵀ, kt .* k_scale, acc)
        i += 1i32
    end
    delta = beta .* (v .- acc)

    # Pass 2: rank-1 update + output
    output = ct.zeros((Dv,), Float32)
    i = 1i32
    while i <= num
        s = ct.load(S, (i, 1, h_v, b), (BLOCK_DK, Dv); padding_mode) → Float32
        kt = ct.load(K_raw, (i, h_k, b), (BLOCK_DK,); padding_mode) → Float32
        qt = ct.load(Q_raw, (i, h_k, b), (BLOCK_DK,); padding_mode) → Float32
        s = s .+ (kt .* k_scale) .* permutedims(delta)
        ct.store(S, (i, 1, h_v, b), s → eltype(S))
        output = mva((s)ᵀ, qt .* q_scale, output)
        i += 1i32
    end

    # RMSNorm + z-gate
    rms_sq = sum(output .* output; dims=1) ./ Float32(Dv)
    rstd = 1.0f0 ./ sqrt.(rms_sq .+ NormEps)
    norm_w = ct.load(NormW, (1,), (Dv,)) → Float32
    z = ct.load(Z, (1, h_v, b), (Dv,)) → Float32
    z_silu = z .* (1.0f0 ./ (1.0f0 .+ exp.(0.0f0 .- z)))
    final = output .* rstd .* norm_w .* z_silu

    ct.store(O, (1, h_v, b), final → OUT_T)
    return
end

function fused_deltanet_decode_step!(O, Q_raw, K_raw, V, Alpha, BetaRaw, Z,
                                     A_log, DtBias, NormWeight, S;
                                     norm_eps=1f-6, verify=nothing)
    Dk, Hk, B = size(Q_raw)
    Dv, Hv, _ = size(V)
    v_per_k = cld(Hv, Hk)
    T = eltype(Q_raw)
    key = (T, Dk, Dv, Hv, Hk)

    function setup()
        saved = copy(S)
        () -> copy!(S, saved)
    end

    autotune_launch(fused_deltanet_decode_fwd,
        CartesianSpace(BLOCK_DK=(8, 16, 32)),
        cfg -> (Hv, B),  # grid over value heads
        cfg -> (
            Q_raw, K_raw, V, Alpha, BetaRaw, Z,
            A_log, DtBias, NormWeight, S, O,
            Constant(Dk), Constant(Dv), Constant(Float32(norm_eps)),
            Constant(cfg.BLOCK_DK), Constant(v_per_k), Constant(T),
        );
        key, verify, setup
    )
end
