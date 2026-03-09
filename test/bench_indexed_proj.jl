using Onion, CUDA
import Onion: cuTileBackend

const ext = Base.get_extension(Onion, :cuTileExt)

# ── Sizes ───────────────────────────────────────────────────────────
D, E, H, k, L = 3072, 128, 128, 24, 8192

# ── Setup ───────────────────────────────────────────────────────────
w = CUDA.randn(Float32, D, E, H)
z = CUDA.randn(Float32, E, k, L)
x = CUDA.randn(Float32, D, L)
w_in = CUDA.randn(Float32, E, D, H)
i_gpu = cu(Int32.(rand(1:H, k, L)))

Z_flat = reshape(z, E, k * L)

# ── 1. Dispatch only (CPU sort + upload) ────────────────────────────
println("=== dispatch only ===")
for _ in 1:3
    CUDA.@time begin
        local d = ext._indexed_proj_dispatch(w, i_gpu, 64)
    end
end

# ── 2. Kernel only (pre-dispatched, atomic scatter to L) ──────────
println("\n=== gather_proj kernel only (atomic scatter to L) ===")
d = ext._indexed_proj_dispatch(w, i_gpu, 64)
O = CUDA.fill!(similar(z, Float32, D, L), 0)
for _ in 1:3
    fill!(O, 0)
    CUDA.@time CUDA.@sync ext.indexed_proj!(
        w, Z_flat, O,
        d.sorted_ids_d, d.sorted_head_ids_d;
        TILE_M=64, scatter_mode=1, gather_mode=0, k_slots=k,
        use_atomic=1)
end

# ── 3. Full gather_proj pipeline ────────────────────────────────────
println("\n=== gather_proj full pipeline ===")
for _ in 1:3
    CUDA.@time CUDA.@sync Onion.gather_proj(cuTileBackend(), w, z, i_gpu)
end

# ── 4. Matmul baseline ─────────────────────────────────────────────
println("\n=== matmul baseline D×(E*k) by (E*k)×L ===")
A = CUDA.randn(Float32, D, E * k)
B = CUDA.randn(Float32, E * k, L)
for _ in 1:3
    CUDA.@time CUDA.@sync A * B
end

# ── 5. scatter_proj kernel only ─────────────────────────────────────
println("\n=== scatter_proj kernel only ===")
d_s = ext._indexed_proj_dispatch(w_in, i_gpu, 64)
Y_flat = CUDA.fill!(similar(x, Float32, E, k * L), 0)
for _ in 1:3
    fill!(Y_flat, 0)
    CUDA.@time CUDA.@sync ext.indexed_proj!(
        w_in, x, Y_flat,
        d_s.sorted_ids_d, d_s.sorted_head_ids_d;
        TILE_M=64, scatter_mode=0, gather_mode=1, k_slots=k)
end

# ── 6. Full scatter_proj pipeline ───────────────────────────────────
println("\n=== scatter_proj full pipeline ===")
for _ in 1:3
    CUDA.@time CUDA.@sync Onion.scatter_proj(cuTileBackend(), w_in, x, i_gpu)
end
