# Full-sequence causal conv1d kernel.
# Grid: (ceil(D/TILE_D), B), each block handles TILE_D channels for all T positions.

function causal_conv1d_sequence_fwd(
    X::TileArray3,       # (D, T, B)
    Weight::TileMatrix,  # (D, K)
    Bias::TileVector,    # (D,) — ignored if HAS_BIAS=false
    Y::TileArray3,       # (D, T, B) — output
    T_len::Int,
    TILE_D::Int,
    KERNEL_SIZE::Int,
    HAS_BIAS::Bool,
    APPLY_SILU::Bool,
)
    padding_mode = ct.PaddingMode.Zero
    pid = ct.bid(1)  # tile along D
    b   = ct.bid(2)  # batch

    # Preload weights for all kernel positions
    # weight[:, k] for k in 1:KERNEL_SIZE
    # We'll accumulate in the inner loop

    bias_tile = HAS_BIAS ?
        ct.load(Bias, (pid,), (TILE_D,); padding_mode) → Float32 :
        ct.zeros((TILE_D,), Float32)

    for t in 1i32:T_len
        acc = ct.zeros((TILE_D,), Float32)

        # Convolve: sum over kernel window
        for k in 1i32:KERNEL_SIZE
            # Source position: t - KERNEL_SIZE + k
            src_t = t - Int32(KERNEL_SIZE) + k
            if src_t >= 1i32
                x_val = ct.load(X, (pid, src_t, b), (TILE_D,); padding_mode) → Float32
            else
                x_val = ct.zeros((TILE_D,), Float32)
            end
            w_val = ct.load(Weight, (pid, k), (TILE_D,); padding_mode) → Float32
            acc = acc .+ x_val .* w_val
        end

        if HAS_BIAS
            acc = acc .+ bias_tile
        end

        if APPLY_SILU
            acc = acc .* (1 ./ (1 .+ exp.(0 .- acc)))
        end

        ct.store(Y, (pid, t, b), acc → eltype(Y))
    end

    return
end

function causal_conv1d_sequence_step!(Y, X, Weight, Bias; silu=true, verify=nothing)
    D, T, B = size(X)
    K = size(Weight, 2)
    has_bias = Bias !== nothing

    key = (eltype(X), D, T, K, has_bias, silu)

    bias_arr = has_bias ? Bias : similar(X, eltype(X), 0)

    autotune_launch(causal_conv1d_sequence_fwd,
        CartesianSpace(TILE_D=(16, 32, 64, 128)),
        cfg -> (cld(D, cfg.TILE_D), B),
        cfg -> (
            X, Weight, bias_arr, Y,
            Constant(T),
            Constant(cfg.TILE_D),
            Constant(K),
            Constant(has_bias),
            Constant(silu),
        );
        key, verify,
    )
end
