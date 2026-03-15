function causal_conv1d_update_fwd(
    X::TileMatrix,       # (D, B) — new input
    State::TileArray3,   # (D, K, B) — ring buffer, mutated
    Weight::TileMatrix,  # (D, K) — conv weights
    Bias::TileVector,    # (D,) — bias (ignored if HAS_BIAS=false)
    Y::TileMatrix,       # (D, B) — output
    TILE_D::Int,
    KERNEL_SIZE::Int,
    HAS_BIAS::Bool,
    APPLY_SILU::Bool,
)
    padding_mode = ct.PaddingMode.Zero
    pid, b = ct.bid(1), ct.bid(2)

    x = ct.load(X, (pid, b), (TILE_D,); padding_mode) → Float32
    acc = ct.zeros((TILE_D,), Float32)

    i = 1i32
    while i <= KERNEL_SIZE
        s = i == KERNEL_SIZE ? x :
            ct.load(State, (pid, i + 1i32, b), (TILE_D,); padding_mode) → Float32

        ct.store(State, (pid, i, b), s → eltype(State))

        w = ct.load(Weight, (pid, i), (TILE_D,); padding_mode) → Float32
        acc = acc .+ s .* w

        i += 1i32
    end

    if HAS_BIAS
        bias = ct.load(Bias, (pid,), (TILE_D,); padding_mode) → Float32
        acc = acc .+ bias
    end

    if APPLY_SILU
        acc = acc .* (1 ./ (1 .+ exp.(0 .- acc)))
    end

    ct.store(Y, (pid, b), acc → eltype(Y))

    return
end

function causal_conv1d_step!(Y, X, State, Weight, Bias;
    silu=true, verify=nothing
)
    D, B = size(X)
    K = size(State, 2)
    has_bias = Bias !== nothing

    key = (eltype(X), D, K, has_bias, silu)

    bias_arr = has_bias ? Bias : similar(X, eltype(X), 0)

    function setup()
        saved = copy(State)
        function reset()
            copy!(State, saved)
        end
    end

    autotune_launch(causal_conv1d_update_fwd,
        CartesianSpace(TILE_D=(16, 32, 64, 128)),
        cfg -> (cld(D, cfg.TILE_D), B),
        cfg -> (
            X, State, Weight, bias_arr, Y,
            Constant(cfg.TILE_D),
            Constant(K),
            Constant(has_bias),
            Constant(silu),
        );
        key, verify, setup
    )
end
