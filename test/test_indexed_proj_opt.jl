using Onion, CUDA
import Onion: cuTileBackend, DefaultBackend

const ext = Base.get_extension(Onion, :cuTileExt)

# ══════════════════════════════════════════════════════════════════════
# Correctness tests for indexed_proj kernels
# ══════════════════════════════════════════════════════════════════════

D, E, H, k, L = 256, 128, 16, 4, 64

w_scatter = CUDA.randn(Float32, E, D, H)
w_gather  = CUDA.randn(Float32, D, E, H)
x         = CUDA.randn(Float32, D, L)
z         = CUDA.randn(Float32, E, k, L)
i_gpu     = cu(Int32.(rand(1:H, k, L)))

# ── Step 1: scatter_proj ──
println("=== Step 1: scatter_proj ===")
y_gpu = Onion.scatter_proj(cuTileBackend(), w_scatter, x, i_gpu)
y_ref = Onion.scatter_proj(DefaultBackend(),
    Float32.(Array(w_scatter)), Float32.(Array(x)), Array(i_gpu))
err = maximum(abs.(Float32.(Array(y_gpu)) .- Float32.(y_ref)))
println("  max err: $err  $(err < 2 ? "PASS" : "FAIL")")

# ── Step 2: gather_proj (high-level, new no-atomics path) ──
println("\n=== Step 2: gather_proj (new path: write kL, sum over k) ===")
o_gpu = Onion.gather_proj(cuTileBackend(), w_gather, z, i_gpu)
o_ref = Onion.gather_proj(DefaultBackend(),
    Float32.(Array(w_gather)), Float32.(Array(z)), Array(i_gpu))
err2 = maximum(abs.(Float32.(Array(o_gpu)) .- Float32.(o_ref)))
println("  max err: $err2  $(err2 < 2 ? "PASS" : "FAIL")")

# ── Step 3: CPU-only sanity check ──
println("\n=== Step 3: CPU-only sanity (DefaultBackend scatter→gather roundtrip) ===")
d_t, e_t, h_t, k_t, l_t = 8, 4, 2, 2, 3
w_in = randn(Float32, e_t, d_t, h_t)
w_out = randn(Float32, d_t, e_t, h_t)
x_t = randn(Float32, d_t, l_t)
i_t = rand(1:h_t, k_t, l_t)
y_t = Onion.scatter_proj(DefaultBackend(), w_in, x_t, i_t)
o_t = Onion.gather_proj(DefaultBackend(), w_out, y_t, i_t)
ref = zeros(Float32, d_t, l_t)
for l in 1:l_t, j in 1:k_t
    ref[:, l] += w_out[:, :, i_t[j, l]] * (w_in[:, :, i_t[j, l]] * x_t[:, l])
end
err_cpu = maximum(abs.(o_t .- ref))
println("  max err: $err_cpu  $(err_cpu < 1e-5 ? "PASS" : "FAIL")")

# ── Step 4: Benchmarks at full size ──
println("\n=== Benchmarks (D=3072, E=128, H=128, k=24, L=8192) ===")
D_f, E_f, H_f, k_f, L_f = 3072, 128, 128, 24, 8192
w_f = CUDA.randn(Float32, D_f, E_f, H_f)
z_f = CUDA.randn(Float32, E_f, k_f, L_f)
i_f = cu(Int32.(rand(1:H_f, k_f, L_f)))

println("\ngather_proj full pipeline:")
for _ in 1:3
    CUDA.@time CUDA.@sync Onion.gather_proj(cuTileBackend(), w_f, z_f, i_f)
end

println("\nscatter_proj full pipeline:")
w_sf = CUDA.randn(Float32, E_f, D_f, H_f)
x_f = CUDA.randn(Float32, D_f, L_f)
for _ in 1:3
    CUDA.@time CUDA.@sync Onion.scatter_proj(cuTileBackend(), w_sf, x_f, i_f)
end

println("\nmatmul baseline D×(E*k) × (E*k)×L:")
A = CUDA.randn(Float32, D_f, E_f * k_f)
B = CUDA.randn(Float32, E_f * k_f, L_f)
for _ in 1:3
    CUDA.@time CUDA.@sync A * B
end

println("\nDone.")
