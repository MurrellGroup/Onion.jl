using NNlib: swish, sigmoid
using Einops: rearrange, @einops_str

# ──── DeltaNet ────

"""
    DeltaNet(in_dim, n_k_heads, n_v_heads; head_dim=128, kernel_size=4)

Gated DeltaNet recurrent layer, as used in Qwen3.5-27B.

Replaces softmax attention with a gated linear recurrence:
- State S ∈ R^{head_dim × head_dim} per head
- Per-token: S = exp(gate) * S + beta * (k ⊗ delta), output = S^T q
- Preceded by causal conv1d on projected QKV

# Examples

```julia
layer = DeltaNet(5120, 16, 48; head_dim=128)
cache = deltanet_cache(layer, 1)
x = randn(Float32, 5120, 1)  # single token decode
output = layer(x; cache)
```
"""
@concrete struct DeltaNet <: Layer
    # Projections
    wqk         # Linear: in_dim => n_k_heads * head_dim * 2 (shared Q/K)
    wv          # Linear: in_dim => n_v_heads * head_dim
    w_alpha     # Linear: in_dim => n_v_heads (gate projection)
    w_beta      # Linear: in_dim => n_v_heads (beta projection)
    wo          # Linear: n_v_heads * head_dim => in_dim

    # Conv1d
    conv_weight # (D_conv, kernel_size) where D_conv = 2*n_k_heads*head_dim + n_v_heads*head_dim
    conv_bias   # (D_conv,) or false

    # Learned parameters
    A_log       # (n_v_heads,) — log gate parameter
    dt_bias     # (n_v_heads,) — dt bias

    # Output norm
    norm        # RMSNorm(head_dim)

    # Config
    head_dim::Int
    n_k_heads::Int
    n_v_heads::Int
    kernel_size::Int
end

function DeltaNet(
    in_dim::Int, n_k_heads::Int, n_v_heads::Int;
    head_dim::Int = 128,
    kernel_size::Int = 4,
    T = Float32,
)
    @assert n_v_heads % n_k_heads == 0 "n_v_heads must be divisible by n_k_heads"

    k_dim = n_k_heads * head_dim
    v_dim = n_v_heads * head_dim
    d_conv = 2 * k_dim + v_dim

    wqk = Linear(in_dim => 2 * k_dim, bias=false)
    wv = Linear(in_dim => v_dim, bias=false)
    w_alpha = Linear(in_dim => n_v_heads, bias=false)
    w_beta = Linear(in_dim => n_v_heads, bias=false)
    wo = Linear(v_dim => in_dim, bias=false)

    conv_weight = randn(T, d_conv, kernel_size) .* T(0.02)
    conv_bias = zeros(T, d_conv)

    A_log = randn(T, n_v_heads) .* T(0.1)
    dt_bias = zeros(T, n_v_heads)

    norm = RMSNorm(head_dim)

    return DeltaNet(
        wqk, wv, w_alpha, w_beta, wo,
        conv_weight, conv_bias,
        A_log, dt_bias,
        norm,
        head_dim, n_k_heads, n_v_heads, kernel_size,
    )
end

function (layer::DeltaNet)(x::AbstractArray{T}; cache) where T
    (; head_dim, n_k_heads, n_v_heads) = layer
    v_per_k = n_v_heads ÷ n_k_heads

    # Project
    qk_proj = layer.wqk(x)      # (2*k_dim, B)
    v_proj = layer.wv(x)         # (v_dim, B)
    alpha = layer.w_alpha(x)     # (n_v_heads, B)
    beta_raw = layer.w_beta(x)   # (n_v_heads, B)

    # Concatenate for conv1d: [q, k, v]
    conv_input = vcat(qk_proj, v_proj) # (D_conv, B)

    # Causal conv1d + SiLU
    conv_out, _ = causal_conv1d(conv_input, cache.conv_state, layer.conv_weight, layer.conv_bias; silu=true)

    # Split back into Q, K, V
    k_dim = n_k_heads * head_dim
    v_dim = n_v_heads * head_dim
    q_raw = conv_out[1:k_dim, :]
    k_raw = conv_out[k_dim+1:2*k_dim, :]
    v_flat = conv_out[2*k_dim+1:end, :]

    # Reshape to per-head: (head_dim, n_heads, B)
    q_heads = rearrange(q_raw, einops"(d h) b -> d h b"; d=head_dim)
    k_heads = rearrange(k_raw, einops"(d h) b -> d h b"; d=head_dim)
    v_heads = rearrange(v_flat, einops"(d h) b -> d h b"; d=head_dim)

    # Repeat interleave Q, K from n_k_heads to n_v_heads
    if v_per_k > 1
        q_heads = repeat(q_heads, einops"d h b -> d (r h) b"; r=v_per_k)
        k_heads = repeat(k_heads, einops"d h b -> d (r h) b"; r=v_per_k)
    end

    # L2 normalize Q and K
    q_norm = sqrt.(sum(q_heads.^2, dims=1) .+ T(1e-6))
    q_heads = q_heads ./ q_norm .* T(1 / sqrt(head_dim))
    k_norm = sqrt.(sum(k_heads.^2, dims=1) .+ T(1e-6))
    k_heads = k_heads ./ k_norm

    # Compute gate: g = -exp(A_log) * softplus(alpha + dt_bias)
    neg_a_exp = .-exp.(layer.A_log)
    sp_input = alpha .+ layer.dt_bias
    sp = @. ifelse(sp_input > T(20), sp_input, log1p(exp(sp_input))) # softplus
    gate = neg_a_exp .* sp  # (n_v_heads, B) — negative, so exp(gate) < 1

    # Compute beta: sigmoid(beta_raw)
    beta = sigmoid.(beta_raw)  # (n_v_heads, B)

    # Recurrent step
    output, _ = deltanet_recurrent(q_heads, k_heads, v_heads, beta, gate, cache.recurrent_state)

    # Output: RMSNorm per head, then multiply by SiLU(gate_z)
    # Note: In the full model, there's a separate gate value from the projection.
    # For simplicity, we apply RMSNorm and reshape to flat.
    output = layer.norm(output)

    # Reshape back: (head_dim, n_v_heads, B) -> (v_dim, B)
    output = rearrange(output, einops"d h b -> (d h) b")

    return layer.wo(output)
end


# ──── DeltaNetCache ────

struct DeltaNetCache{T, CS<:AbstractArray{T,3}, RS<:AbstractArray{T,4}}
    conv_state::CS       # (D_conv, kernel_size, B)
    recurrent_state::RS  # (Dk, Dv, H, B)
end

function deltanet_cache(layer::DeltaNet, batch::Int=1)
    (; head_dim, n_k_heads, n_v_heads, kernel_size) = layer
    d_conv = 2 * n_k_heads * head_dim + n_v_heads * head_dim
    conv_state = zeros_like(layer.conv_weight, d_conv, kernel_size, batch)
    recurrent_state = zeros_like(layer.wv.weight, head_dim, head_dim, n_v_heads, batch)
    return DeltaNetCache(conv_state, recurrent_state)
end
