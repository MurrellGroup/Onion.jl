using CUDA
using cuTile
using Onion

Onion.backend!(cuTileBackend())

function bench_block_sparse_proj(;
    D = 3072, E = 128, H = 96, k = 24, L = 8192,
    T = Float32, warmup = 3, nruns = 10,
)
    # Random routing
    i = Int32.(rand(1:H, k, L))

    W_in  = CUDA.rand(T, E, D, H)
    W_out = CUDA.rand(T, D, E, H)
    X     = CUDA.rand(T, D, L)

    println("Config: D=$D, E=$E, H=$H, k=$k, L=$L, T=$T")
    println("  W_in:  $(size(W_in))  W_out: $(size(W_out))  X: $(size(X))")
    println("  k*L = $(k*L) flat entries")
    println()

    # ── In-proj forward ──────────────────────────────────────────────
    Y = nothing
    CUDA.@sync for _ in 1:warmup
        Y = Onion.block_sparse_in_proj(W_in, X, i)
    end
    times_in = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed begin
            Y = Onion.block_sparse_in_proj(W_in, X, i)
        end
        push!(times_in, t * 1000)
    end
    println("In-proj fwd:   median=$(round(median(times_in); digits=3)) ms  min=$(round(minimum(times_in); digits=3)) ms")

    # ── Out-proj forward ─────────────────────────────────────────────
    Z = nothing
    CUDA.@sync for _ in 1:warmup
        Z = Onion.block_sparse_out_proj(W_out, Y, i)
    end
    times_out = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed begin
            Z = Onion.block_sparse_out_proj(W_out, Y, i)
        end
        push!(times_out, t * 1000)
    end
    println("Out-proj fwd:  median=$(round(median(times_out); digits=3)) ms  min=$(round(minimum(times_out); digits=3)) ms")

    # ── In-proj + Out-proj combined ──────────────────────────────────
    CUDA.@sync for _ in 1:warmup
        Y2 = Onion.block_sparse_in_proj(W_in, X, i)
        Z2 = Onion.block_sparse_out_proj(W_out, Y2, i)
    end
    times_total = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed begin
            Y2 = Onion.block_sparse_in_proj(W_in, X, i)
            Z2 = Onion.block_sparse_out_proj(W_out, Y2, i)
        end
        push!(times_total, t * 1000)
    end
    println("Total fwd:     median=$(round(median(times_total); digits=3)) ms  min=$(round(minimum(times_total); digits=3)) ms")

    # ── Correctness check ────────────────────────────────────────────
    Y_cpu = Array(Y)
    X_cpu = Array(X)
    W_in_cpu = Array(W_in)
    W_out_cpu = Array(W_out)

    Y_ref = zeros(T, E, k * L)
    for tok in 1:L, slot in 1:k
        h = i[slot, tok]
        flat = (tok - 1) * k + slot
        Y_ref[:, flat] = W_in_cpu[:, :, h] * X_cpu[:, tok]
    end

    Z_ref = zeros(T, D, L)
    for tok in 1:L, slot in 1:k
        h = i[slot, tok]
        flat = (tok - 1) * k + slot
        Z_ref[:, tok] += W_out_cpu[:, :, h] * Y_ref[:, flat]
    end

    y_err = maximum(abs.(Y_cpu .- Y_ref))
    z_err = maximum(abs.(Array(Z) .- Z_ref))
    println("\nMax error:  Y=$(y_err)  Z=$(z_err)")
    println("Correct:   Y=$(isapprox(Y_cpu, Y_ref; rtol=1e-2))  Z=$(isapprox(Array(Z), Z_ref; rtol=1e-2))")

    return (; times_in, times_out, times_total, Y, Z)
end

using Statistics: median

bench_block_sparse_proj()
