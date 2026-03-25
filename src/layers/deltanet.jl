using NNlib: swish, sigmoid
using Einops: rearrange, @einops_str

# ──── DeltaNet ────

"""
    DeltaNet(hidden_size, n_k_heads, n_v_heads; head_dim=128, kernel_size=4)

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
@kwdef @concrete struct DeltaNet <: Layer
    hidden_size::Int
    n_k_heads::Int
    n_v_heads::Int   = n_k_heads
    head_dim::Int    = 128 # TODO: differentiate between key and value head dim
    kernel_size::Int = 4
    T::Type          = Float32

    q_dim::Int  = n_k_heads * head_dim
    k_dim::Int  = n_k_heads * head_dim
    v_dim::Int  = n_v_heads * head_dim
    d_conv::Int = 2 * k_dim + v_dim

    qk_proj = Linear(hidden_size => n_k_heads * head_dim * 2)
    v_proj  = Linear(hidden_size => n_v_heads * head_dim)
    alpha_proj = Linear(hidden_size => n_v_heads)
    beta_proj  = Linear(hidden_size => n_v_heads)
    o_proj  = Linear(n_v_heads * head_dim => hidden_size)

    conv_weight = randn(T, d_conv, kernel_size) .* T(0.02)
    conv_bias   = zeros(T, d_conv)

    A_log       = randn(T, n_v_heads) .* T(0.1)
    dt_bias     = zeros(T, n_v_heads)

    norm        = RMSNorm(head_dim)
end

function decode(layer::DeltaNet,
    x::AbstractArray{T};
    cache
) where T
    b = backend()
    (; head_dim, q_dim, k_dim, v_dim) = layer

    qk_proj = layer.qk_proj(x)       # (2*k_dim, B)
    v_proj = layer.v_proj(x)         # (v_dim, B)
    alpha = layer.alpha_proj(x)     # (n_v_heads, B)
    beta_raw = layer.beta_proj(x)   # (n_v_heads, B)

    conv_out, _ = causal_conv1d(b, [qk_proj; v_proj], cache.conv_state, layer.conv_weight, layer.conv_bias; silu=true)

    q, k, v = splitaxis(conv_out, (q_dim, k_dim, v_dim))

    qʰ, kʰ, vʰ = rearrange.((q, k, v), einops"(d h) ... -> d h ..."; d=head_dim)

    qʰ = qʰ ./ .√(sum(abs2, qʰ, dims=1) .+ T(1e-6)) .* T(1 / sqrt(head_dim))
    kʰ = kʰ ./ .√(sum(abs2, kʰ, dims=1) .+ T(1e-6))

    sp_input = alpha .+ layer.dt_bias
    sp = @. ifelse(sp_input > T(20), sp_input, log1p(exp(sp_input))) # softplus
    gate = .-exp.(layer.A_log) .* sp  # (n_v_heads, B) — negative, so exp(gate) < 1

    β = sigmoid.(beta_raw)  # (n_v_heads, B)

    output, _ = deltanet_recurrent_decode(b, qʰ, kʰ, vʰ, β, gate, cache.recurrent_state)

    # Output: RMSNorm per head, then multiply by SiLU(gate_z)
    # Note: In the full model, there's a separate gate value from the projection.
    # For simplicity, we apply RMSNorm and reshape to flat.
    output = layer.norm(output)

    output = rearrange(output, einops"d h ... -> (d h) ...")

    return layer.o_proj(output)
end


function forward(layer::DeltaNet,
    x::AbstractArray{T};
    cache = nothing,
) where T
    b = backend(r)
    (; head_dim, q_dim, k_dim, v_dim, n_v_heads) = layer

    qk_proj = layer.qk_proj(x)       # (2*k_dim, L, ...)
    v_proj  = layer.v_proj(x)         # (v_dim, L, ...)
    alpha   = layer.alpha_proj(x)     # (n_v_heads, L, ...)
    beta_raw = layer.beta_proj(x)     # (n_v_heads, L, ...)

    # Causal conv1d over the full sequence
    qkv = cat(qk_proj, v_proj; dims=1)
    conv_out = causal_conv1d_sequence(b, qkv, layer.conv_weight, layer.conv_bias; silu=true)

    q, k, v = splitaxis(conv_out, (q_dim, k_dim, v_dim))

    # Reshape to per-head: (D, L, ...) → (Dh, L, H, ...)
    qʰ, kʰ, vʰ = rearrange.((q, k, v), einops"(d h) l ... -> d l h ..."; d=head_dim)

    # L2 normalize Q and K (over head dim, i.e. dim=1)
    qʰ = qʰ ./ .√(sum(abs2, qʰ; dims=1) .+ T(1e-6)) .* T(1 / sqrt(head_dim))
    kʰ = kʰ ./ .√(sum(abs2, kʰ; dims=1) .+ T(1e-6))

    # Gate and beta
    sp_input = alpha .+ layer.dt_bias
    sp = @. ifelse(sp_input > T(20), sp_input, log1p(exp(sp_input)))
    gate = .-exp.(layer.A_log) .* sp
    β = sigmoid.(beta_raw)

    # Full-sequence recurrence
    initial_state = isnothing(cache) ? nothing : cache.recurrent_state
    output, final_state = deltanet_sequence(b, qʰ, kʰ, vʰ, β, gate, initial_state)

    # Write final state back to cache if provided
    if !isnothing(cache)
        cache.recurrent_state .= final_state
        # Update conv state to last K-1 positions for decode continuation
        K = size(layer.conv_weight, 2)
        L = size(qkv, 2)
        if L >= K
            cache.conv_state .= qkv[:, L-K+1:L, size(qkv)[3:end]...]
        end
    end

    # RMSNorm per head
    output = layer.norm(output)

    output = rearrange(output, einops"d l h ... -> (d h) l ...")

    return layer.o_proj(output)
end

function deltanet_cache(layer::DeltaNet, batch::Int=1)
    (; head_dim, n_k_heads, n_v_heads, kernel_size) = layer
    d_conv = 2 * n_k_heads * head_dim + n_v_heads * head_dim
    conv_state = zeros_like(layer.conv_weight, d_conv, kernel_size, batch)
    recurrent_state = zeros_like(layer.v_proj.weight, head_dim, head_dim, n_v_heads, batch)
    return (; conv_state, recurrent_state)
end
