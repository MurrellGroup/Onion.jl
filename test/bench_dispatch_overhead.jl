using CUDA, cuTile, Onion
using Statistics: median

Onion.backend!(cuTileBackend())

D = 3072; E = 128; H = 96; k = 24; L = 8192; T = Float32
TILE_M = 64

i = Int32.(rand(1:H, k, L))
W_in  = CUDA.rand(T, E, D, H)
W_out = CUDA.rand(T, D, E, H)
X     = CUDA.rand(T, D, L)

# ── CPU dispatch build time ──────────────────────────────────────────
println("=== Dispatch build (CPU) ===")
# warmup
Onion.block_sparse_proj_dispatch(i, H, TILE_M)

times_dispatch = Float64[]
for _ in 1:20
    t = @elapsed Onion.block_sparse_proj_dispatch(i, H, TILE_M)
    push!(times_dispatch, t * 1000)
end
println("  median=$(round(median(times_dispatch); digits=3)) ms  min=$(round(minimum(times_dispatch); digits=3)) ms")

# ── Device upload time ───────────────────────────────────────────────
println("\n=== Device upload ===")
d = Onion.block_sparse_proj_dispatch(i, H, TILE_M)

function upload_all(d, ref)
    _to_dev(v) = copyto!(similar(ref, eltype(v), size(v)), v)
    sorted_ids_d    = _to_dev(d.sorted_ids)
    tok_cols_d      = _to_dev(d.tok_cols)
    block_heads_d   = _to_dev(d.block_heads)
    hbs_d           = _to_dev(d.head_block_starts)
    slot_devs = map(d.slot_dispatches) do sd
        (; sorted_tok_ids = _to_dev(sd.sorted_tok_ids),
           sorted_flat_ids = _to_dev(sd.sorted_flat_ids),
           block_heads = _to_dev(sd.block_heads))
    end
    return sorted_ids_d, tok_cols_d, block_heads_d, hbs_d, slot_devs
end

# warmup
upload_all(d, X)

times_upload = Float64[]
for _ in 1:20
    CUDA.@sync upload_all(d, X)
    t = CUDA.@elapsed upload_all(d, X)
    push!(times_upload, t * 1000)
end
println("  median=$(round(median(times_upload); digits=3)) ms  min=$(round(minimum(times_upload); digits=3)) ms")

# ── Dispatch table sizes ─────────────────────────────────────────────
println("\n=== Dispatch table sizes ===")
println("  flat sorted_ids: $(length(d.sorted_ids)) entries ($(sizeof(d.sorted_ids) ÷ 1024) KiB)")
println("  flat tok_cols:   $(length(d.tok_cols)) entries")
println("  block_heads:     $(length(d.block_heads)) blocks")
for (s, sd) in enumerate(d.slot_dispatches)
    if s == 1
        println("  slot $s: sorted_tok_ids=$(length(sd.sorted_tok_ids)), sorted_flat_ids=$(length(sd.sorted_flat_ids)), blocks=$(length(sd.block_heads))")
    end
end
println("  ($(k) slots total, all similar size)")
let tb = sizeof(d.sorted_ids) + sizeof(d.tok_cols) + sizeof(d.block_heads) + sizeof(d.head_block_starts)
    for sd in d.slot_dispatches
        tb += sizeof(sd.sorted_tok_ids) + sizeof(sd.sorted_flat_ids) + sizeof(sd.block_heads)
    end
    println("  total dispatch data: $(tb ÷ 1024) KiB")
end

# ── Full end-to-end (dispatch + upload + kernel) ─────────────────────
println("\n=== End-to-end (dispatch + upload + kernel) ===")
# warmup
CUDA.@sync for _ in 1:3
    Y = Onion.block_sparse_in_proj(W_in, X, i)
    Z = Onion.block_sparse_out_proj(W_out, Y, i)
end

times_e2e = Float64[]
for _ in 1:10
    t = CUDA.@elapsed begin
        Y = Onion.block_sparse_in_proj(W_in, X, i)
        Z = Onion.block_sparse_out_proj(W_out, Y, i)
    end
    push!(times_e2e, t * 1000)
end
println("  median=$(round(median(times_e2e); digits=3)) ms  min=$(round(minimum(times_e2e); digits=3)) ms")

println("\n=== Overhead breakdown ===")
dispatch_ms = median(times_dispatch)
upload_ms = median(times_upload)
e2e_ms = median(times_e2e)
routing_ms = 2 * dispatch_ms + 2 * upload_ms  # dispatch called twice (in + out)
kernel_est = e2e_ms - routing_ms
println("  CPU dispatch (×1):  $(round(dispatch_ms; digits=3)) ms")
println("  Device upload (×1): $(round(upload_ms; digits=3)) ms")
println("  Routing total (×2): $(round(routing_ms; digits=3)) ms")
println("  End-to-end:         $(round(e2e_ms; digits=3)) ms")
println("  Kernel est:         $(round(kernel_est; digits=3)) ms")
println("  Routing overhead:   $(round(routing_ms / e2e_ms * 100; digits=1))%")
