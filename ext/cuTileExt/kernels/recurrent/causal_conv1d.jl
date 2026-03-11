# Causal depthwise conv1d state update kernel (decode step)
#
# Translated from triton_causal_conv1d.py (multi-channel variant).
# Each block processes TILE_D channels:
#   1. Shift state left by 1
#   2. Insert new input at the end
#   3. Dot product with weights
#   4. Optional bias + SiLU activation
#
# Grid: (ceil(D/TILE_D), B) — multi-channel per block
# Conv state is [D, K, B] mutated in-place

function causal_conv1d_update_fwd(
    X::TileMatrix,       # (D, B) — new input
    State::TileArray3,   # (D, K, B) — ring buffer, mutated
    Weight::TileMatrix,  # (D, K) — conv weights
    Bias::TileVector,    # (D,) — bias (ignored if HAS_BIAS=false)
    Y::TileMatrix,       # (D, B) — output
    D::Int,
    TILE_D::Int,
    KERNEL_SIZE::Int,
    HAS_BIAS::Bool,
    APPLY_SILU::Bool,
)
    pid = ct.bid(1)    # channel block index
    b = ct.bid(2)      # batch

    # Load input for TILE_D channels
    x_val = ct.load(X, (pid, b), (TILE_D, 1); padding_mode=ct.PaddingMode.Zero) → Float32
    x_val = reshape(x_val, (TILE_D,))

    acc = ct.zeros((TILE_D,), Float32)

    # Process each kernel position: shift left + convolve
    # Position 1..KERNEL_SIZE-1: load from next position (shift left)
    # Position KERNEL_SIZE: insert new input
    i = 1i32
    while i <= Int32(KERNEL_SIZE)
        if i < Int32(KERNEL_SIZE)
            s_val = ct.load(State, (pid, i + 1i32, b), (TILE_D, 1, 1); padding_mode=ct.PaddingMode.Zero) → Float32
        else
            s_val = reshape(x_val, (TILE_D, 1, 1))
        end

        # Store shifted value
        ct.store(State, (pid, i, b), s_val → eltype(State))

        s_val = reshape(s_val, (TILE_D,))

        # Load weight and accumulate
        w_val = ct.load(Weight, (pid, i), (TILE_D, 1); padding_mode=ct.PaddingMode.Zero) → Float32
        w_val = reshape(w_val, (TILE_D,))
        acc = acc .+ s_val .* w_val

        i += 1i32
    end

    # Bias
    if HAS_BIAS
        bias_val = ct.load(Bias, (pid,), (TILE_D,); padding_mode=ct.PaddingMode.Zero) → Float32
        acc = acc .+ bias_val
    end

    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    if APPLY_SILU
        acc = acc .* (1f0 ./ (1f0 .+ exp.(0f0 .- acc)))
    end

    ct.store(Y, (pid, b), reshape(acc → eltype(Y), (TILE_D, 1)))

    return
end

function causal_conv1d_step!(Y, X, State, Weight, Bias;
    silu=true, verify=nothing,
)
    D, B = size(X)
    K = size(State, 2)
    has_bias = Bias !== nothing

    key = (eltype(X), D, K, has_bias, silu)

    bias_arr = has_bias ? Bias : similar(X, eltype(X), 0)

    autotune_launch(causal_conv1d_update_fwd,
        CartesianSpace(TILE_D=(16, 32, 64, 128)),
        cfg -> (cld(D, cfg.TILE_D), B),
        cfg -> (
            X, State, Weight, bias_arr, Y,
            Constant(D),
            Constant(cfg.TILE_D),
            Constant(K),
            Constant(has_bias),
            Constant(silu),
        );
        key, verify
    )
end
