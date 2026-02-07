# AnyGPUArray dispatch overrides for protein layer hooks.
# Routes to ONIONop KA kernels for GPU acceleration on any GPU backend.
# OnionTile (CuArray-specific) overrides take priority when loaded.

using GPUArraysCore: AnyGPUArray
using ONIONop: ONIONop

# ============================================================================
# Helper: copy output into pre-allocated buffer if provided
# ============================================================================

function _maybe_copy_output(o, output)
    if output !== nothing
        copyto!(output, o)
        return output
    end
    return o
end

# ============================================================================
# LayerNorm dispatch → ONIONop.layer_norm
# CRITICAL: ONIONop defaults ϵ=1f-6 but LayerNormFirst uses eps=1f-5.
# Both use PyTorch convention: (x - μ) / sqrt(var + eps) * w + b
# with population variance (divide by N, not N-1).
# ============================================================================

function layernorm_first_forward(x::AnyGPUArray, w::AnyGPUArray, b::AnyGPUArray; eps=1f-5)
    x2d = reshape(x, size(x, 1), :)
    y = ONIONop.layer_norm(x2d, vec(w), vec(b); ϵ=Float32(eps))
    return reshape(y, size(x))
end

# ============================================================================
# Flash attention (no bias) → ONIONop.flash_attention
# ONIONop computes scale internally as 1/sqrt(emb_dim).
# In (D,L,H,B) layout, emb_dim == head_width, so scale matches Onion's.
# ============================================================================

function flash_attention_forward(Q::AnyGPUArray{T,4}, K::AnyGPUArray{T,4}, V::AnyGPUArray{T,4};
                                  scale=nothing, output=nothing) where T
    o = ONIONop.flash_attention(Q, K, V, nothing; causal=false)
    return _maybe_copy_output(o, output)
end

# ============================================================================
# Flash attention (with bias) → ONIONop.flash_attention
# Onion passes bias as (K, Q, H, B); ONIONop expects pair as (H, Q, K, B).
# ============================================================================

function flash_attention_bias_forward(Q::AnyGPUArray{T,4}, K::AnyGPUArray{T,4}, V::AnyGPUArray{T,4},
                                       bias::AnyGPUArray; scale=nothing, output=nothing) where T
    # (K, Q, H, B) → (H, Q, K, B)
    pair = permutedims(bias, (3, 2, 1, 4))

    # Expand pair batch dim to match Q when needed.
    # Triangle attention flattens (B, L) → B*L via column-major reshape,
    # so B_q = B_pair * L.  Julia's repeat tiles correctly for this pattern:
    # repeat(pair, 1,1,1, stride) maps index i → pair[:,:,:, ((i-1) % B_pair) + 1].
    B_q = size(Q, 4)
    B_p = size(pair, 4)
    if B_p != B_q
        @assert B_q % B_p == 0 "Q batch ($B_q) must be divisible by pair batch ($B_p)"
        pair = repeat(pair, 1, 1, 1, B_q ÷ B_p)
    end

    o = ONIONop.flash_attention(Q, K, V, pair; causal=false)
    return _maybe_copy_output(o, output)
end

# ============================================================================
# Rotary positional embeddings
# ONIONop.llama_rope rotates q+k jointly — not suitable for ESMFold's
# single-array rotation. Broadcasting-based _cpu_rotary_pos_emb works
# on any GPU array automatically via GPUArraysCore dispatch.
# ============================================================================

function rotary_pos_emb_forward(x::AnyGPUArray, cos::AbstractArray, sin::AbstractArray)
    return _cpu_rotary_pos_emb(x, cos, sin)
end

# ============================================================================
# Triangle multiplication contraction
# NNlib.batched_mul dispatches to cuBLAS on CUDA, generic GPU kernels
# on other backends. No need for a dedicated KA kernel.
# ============================================================================

function combine_projections_forward(a::AnyGPUArray, b::AnyGPUArray, outgoing::Bool)
    return _cpu_combine_projections(a, b, outgoing)
end
