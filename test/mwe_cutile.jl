# Integration test for cuTile scatter_proj/gather_proj kernels (forward + backward)
# Run with: julia --project=@oniontile test/mwe_cutile.jl

using CUDA
using cuTile
using Onion
using Zygote

d_model, d_head, n_heads, k, L = 64, 32, 8, 4, 10

println("=== scatter_proj / gather_proj cuTile integration test ===")
println("  d_model=$d_model, d_head=$d_head, n_heads=$n_heads, k=$k, L=$L")
println()

# Build deterministic routing indices (CPU for reference, GPU for cuTile)
r = randn(Float32, n_heads, d_model)
x_route = randn(Float32, d_model, L)
scores = r * x_route
i_cpu = mapslices(v -> partialsortperm(v, 1:k, rev=true), scores, dims=1)
i_gpu = CuArray(i_cpu)

# ── scatter_proj: Forward ─────────────────────────────────────────────
println("--- scatter_proj: Forward ---")
w_cpu = randn(Float32, d_head, d_model, n_heads)
x_cpu = randn(Float32, d_model, L)
w = CuArray(w_cpu)
x = CuArray(x_cpu)

ref = Onion.scatter_proj(DefaultBackend(), w, x, i_gpu)
y   = Onion.scatter_proj(cuTileBackend(), w, x, i_gpu)

err = maximum(abs.(Array(y) .- Array(ref)))
println("  maxerr=$err  ", err < 0.01 ? "PASS" : "FAIL")

# ── scatter_proj: Backward ────────────────────────────────────────────
println("--- scatter_proj: Backward ---")
f_sp_cpu(w, x) = sum(Onion.scatter_proj(w, x, i_cpu))
f_sp_gpu(w, x) = sum(Onion.scatter_proj(w, x, i_gpu))

gw_ref, gx_ref = Zygote.gradient(f_sp_cpu, w_cpu, x_cpu)
gw_ct, gx_ct   = Zygote.gradient(f_sp_gpu, w, x)

err_gw = maximum(abs.(Array(gw_ct) .- gw_ref))
err_gx = maximum(abs.(Array(gx_ct) .- gx_ref))
println("  dW maxerr=$err_gw  ", err_gw < 0.1 ? "PASS" : "FAIL")
println("  dX maxerr=$err_gx  ", err_gx < 0.1 ? "PASS" : "FAIL")

# ── gather_proj: Forward ──────────────────────────────────────────────
println("--- gather_proj: Forward ---")
w_out_cpu = randn(Float32, d_model, d_head, n_heads)
z_cpu = randn(Float32, d_head, k, L)
w_out = CuArray(w_out_cpu)
z = CuArray(z_cpu)

ref_g = Onion.gather_proj(DefaultBackend(), w_out, z, i_gpu)
o     = Onion.gather_proj(cuTileBackend(), w_out, z, i_gpu)

err_g = maximum(abs.(Array(o) .- Array(ref_g)))
println("  maxerr=$err_g  ", err_g < 0.01 ? "PASS" : "FAIL")
println("  output shape: $(size(o))  ", size(o) == (d_model, L) ? "PASS" : "FAIL")

# ── gather_proj: Backward ─────────────────────────────────────────────
println("--- gather_proj: Backward ---")
f_gp_cpu(w, z) = sum(Onion.gather_proj(w, z, i_cpu))
f_gp_gpu(w, z) = sum(Onion.gather_proj(w, z, i_gpu))

gw_ref_g, gz_ref_g = Zygote.gradient(f_gp_cpu, w_out_cpu, z_cpu)
gw_ct_g, gz_ct_g   = Zygote.gradient(f_gp_gpu, w_out, z)

err_gw_g = maximum(abs.(Array(gw_ct_g) .- gw_ref_g))
err_gz_g = maximum(abs.(Array(gz_ct_g) .- gz_ref_g))
println("  dW maxerr=$err_gw_g  ", err_gw_g < 0.1 ? "PASS" : "FAIL")
println("  dZ maxerr=$err_gz_g  ", err_gz_g < 0.1 ? "PASS" : "FAIL")

# ── Roundtrip: scatter_proj → gather_proj ─────────────────────────────
println("--- Roundtrip: scatter_proj → gather_proj ---")
f_rt_gpu(w_in, w_out, x) = sum(Onion.gather_proj(w_out, Onion.scatter_proj(w_in, x, i_gpu), i_gpu))
f_rt_cpu(w_in, w_out, x) = sum(Onion.gather_proj(w_out, Onion.scatter_proj(w_in, x, i_cpu), i_cpu))

rt_ref = f_rt_cpu(w_cpu, w_out_cpu, x_cpu)
rt_ct  = f_rt_gpu(w, w_out, x)
rt_err = abs(rt_ct - rt_ref)
println("  fwd err=$rt_err  ", rt_err < 0.1 ? "PASS" : "FAIL")

# ── Determinism: run forward 5 times ─────────────────────────────────
println("--- Determinism (5 runs) ---")
results = [Array(Onion.scatter_proj(cuTileBackend(), w, x, i_gpu)) for _ in 1:5]
all_same = all(r == results[1] for r in results)
println("  All identical: $all_same  ", all_same ? "PASS" : "FAIL")

println()

# ═══════════════════════════════════════════════════════════════════════
# top_multihead_ffn cuTile forward test
# ═══════════════════════════════════════════════════════════════════════
println("=== top_multihead_ffn cuTile forward test ===")

D_mh, I_dim, H_mh = 32, 64, 8
k_mh, L_mh = 4, 10

K_mh_cpu = randn(Float32, I_dim, D_mh, H_mh)
U_mh_cpu = randn(Float32, I_dim, D_mh, H_mh)
V_mh_cpu = randn(Float32, D_mh, I_dim, H_mh)
Q_mh_cpu = randn(Float32, D_mh, k_mh, L_mh)
i_mh_cpu = rand(1:H_mh, k_mh, L_mh)

K_mh = CuArray(K_mh_cpu)
U_mh = CuArray(U_mh_cpu)
V_mh = CuArray(V_mh_cpu)
Q_mh = CuArray(Q_mh_cpu)
i_mh_gpu = CuArray(i_mh_cpu)

ref_mh = Onion.top_multihead_ffn(DefaultBackend(), Q_mh_cpu, K_mh_cpu, U_mh_cpu, V_mh_cpu, Onion.swish, i_mh_cpu)
y_mh   = Onion.top_multihead_ffn(cuTileBackend(), Q_mh, K_mh, U_mh, V_mh, Onion.swish, i_mh_gpu)
y_mh_arr = Array(y_mh)

abs_err_mh = maximum(abs.(y_mh_arr .- ref_mh))
ref_max_mh = maximum(abs.(ref_mh))
rel_pct_mh = 100 * abs_err_mh / ref_max_mh
println("  rel=$(round(rel_pct_mh, digits=3))%  ", rel_pct_mh < 0.1 ? "PASS" : "FAIL")

# ── Determinism ────────────────────────────────────────────────────────
results_mh = [Array(Onion.top_multihead_ffn(cuTileBackend(), Q_mh, K_mh, U_mh, V_mh, Onion.swish, i_mh_gpu)) for _ in 1:5]
all_same_mh = all(r == results_mh[1] for r in results_mh)
println("  Determinism (5 runs): $all_same_mh  ", all_same_mh ? "PASS" : "FAIL")

# ── Edge case: I not divisible by TILE_I ────────────────────────────────
println("--- Edge case: I=48 (not power-of-2 aligned) ---")
I_odd = 48
K_odd_cpu = randn(Float32, I_odd, D_mh, H_mh)
U_odd_cpu = randn(Float32, I_odd, D_mh, H_mh)
V_odd_cpu = randn(Float32, D_mh, I_odd, H_mh)

ref_odd = Onion.top_multihead_ffn(DefaultBackend(), Q_mh_cpu, K_odd_cpu, U_odd_cpu, V_odd_cpu, Onion.swish, i_mh_cpu)
y_odd   = Onion.top_multihead_ffn(cuTileBackend(), CuArray(Q_mh_cpu), CuArray(K_odd_cpu), CuArray(U_odd_cpu), CuArray(V_odd_cpu), Onion.swish, i_mh_gpu)
y_odd_arr = Array(y_odd)

abs_err_odd = maximum(abs.(y_odd_arr .- ref_odd))
ref_max_odd = maximum(abs.(ref_odd))
rel_pct_odd = 100 * abs_err_odd / ref_max_odd
println("  rel=$(round(rel_pct_odd, digits=3))%  ", rel_pct_odd < 0.1 ? "PASS" : "FAIL")

# ── Backward: cuTile rrule vs finite differences ──────────────────────
println("--- Backward: dQ, dK, dU, dV ---")
f_mh_gpu(Q, K, U, V) = sum(Onion.top_multihead_ffn(cuTileBackend(), Q, K, U, V, Onion.swish, i_mh_gpu))
gQ, gK, gU, gV = Zygote.gradient(f_mh_gpu, Q_mh, K_mh, U_mh, V_mh)

# Spot-check dQ against finite differences
function fd_gradient(f, x; eps=1f-3)
    g = similar(x)
    for idx in eachindex(x)
        x_plus = copy(x); x_plus[idx] += eps
        x_minus = copy(x); x_minus[idx] -= eps
        g[idx] = (f(x_plus) - f(x_minus)) / (2 * eps)
    end
    return g
end

fQ_fd(Q) = sum(Onion.top_multihead_ffn(Q, K_mh_cpu, U_mh_cpu, V_mh_cpu, Onion.swish, i_mh_cpu))
gQ_fd = fd_gradient(fQ_fd, Q_mh_cpu)
gQ_err = maximum(abs.(Array(gQ) .- gQ_fd))
gQ_max = maximum(abs.(gQ_fd))
gQ_rel = 100 * gQ_err / gQ_max
println("  dQ: rel=$(round(gQ_rel, digits=3))%  ", gQ_rel < 1.0 ? "PASS" : "FAIL")
println("  shapes: dQ=$(size(gQ)) dK=$(size(gK)) dU=$(size(gU)) dV=$(size(gV))")
println("  no NaN: $(all(!isnan, Array(gQ)) && all(!isnan, Array(gK)) && all(!isnan, Array(gU)) && all(!isnan, Array(gV)))")

println()

# ═══════════════════════════════════════════════════════════════════════
# top_k_gating cuTile test
# ═══════════════════════════════════════════════════════════════════════
println("=== top_k_gating cuTile test ===")

H_g, L_g, k_g = 8, 20, 4
scores_cpu = randn(Float32, H_g, L_g)
scores_gpu = CuArray(scores_cpu)

# Forward
ref_idx, ref_v = Onion.top_k_gating(DefaultBackend(), scores_cpu, k_g)
ct_idx, ct_v = Onion.top_k_gating(cuTileBackend(), scores_gpu, k_g)

idx_match = all(Array(ct_idx) .== ref_idx)
v_err = maximum(abs.(Array(ct_v) .- ref_v))
println("  Indices match: $idx_match  ", idx_match ? "PASS" : "FAIL")
println("  Values err: $v_err  ", v_err < 1e-5 ? "PASS" : "FAIL")

# Backward
c_g = CuArray(reshape(Float32.(1:k_g), k_g, 1))
f_gating(s) = sum(Onion.top_k_gating(cuTileBackend(), s, k_g)[2] .* c_g)
g_ct = Array(Zygote.gradient(f_gating, scores_gpu)[1])

scores_f64 = Float64.(scores_cpu)
c64 = Float64.(Array(c_g))
function fd_gradient_f64(f, x; eps=1e-6)
    g = similar(x)
    for idx in eachindex(x)
        xp = copy(x); xp[idx] += eps
        xm = copy(x); xm[idx] -= eps
        g[idx] = (f(xp) - f(xm)) / (2 * eps)
    end
    return g
end
f_ref(s) = sum(Onion.top_k_gating(s, k_g)[2] .* c64)
g_ref = Float32.(fd_gradient_f64(f_ref, scores_f64))
g_err = maximum(abs.(g_ct .- g_ref))
g_max = maximum(abs.(g_ref))
g_rel = g_max > 0 ? 100 * g_err / g_max : 0.0
println("  Backward rel: $(round(g_rel, digits=3))%  ", g_rel < 0.5 ? "PASS" : "FAIL")

println()
println("=== Done ===")
