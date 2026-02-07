# AnyGPUArray layer overrides for performance-critical protein layers.
# Uses in-place operations (.+=, .=) that work on any GPU backend.
# No buffer pool, no CUDA.unsafe_free! — those are OnionTile-specific.
#
# AD compatibility: uses ONIONop.within_gradient to select in-place (inference)
# vs out-of-place (training) operations. within_gradient(x) returns false
# normally and true inside Zygote pullbacks via its rrule.

using GPUArraysCore: AnyGPUArray
using ONIONop: within_gradient

# ============================================================================
# TriangularSelfAttentionBlock: in-place residual additions
# This is the most important override — 7 residual additions that each
# allocate a new array in the CPU path.
# ============================================================================

function (m::TriangularSelfAttentionBlock)(sequence_state::AnyGPUArray, pairwise_state::AnyGPUArray;
        mask=nothing, chunk_size=nothing, residue_index=nothing)
    bias = m.pair_to_sequence(pairwise_state)
    y = m.layernorm_1(sequence_state)
    y, _ = m.seq_attention(y; mask=mask, bias=bias)

    if within_gradient(sequence_state)
        sequence_state = sequence_state .+ m.drop(y)
    else
        sequence_state .+= m.drop(y)
    end
    sequence_state = m.mlp_seq(sequence_state)

    if within_gradient(pairwise_state)
        pairwise_state = pairwise_state .+ m.sequence_to_pair(sequence_state)
    else
        pairwise_state .+= m.sequence_to_pair(sequence_state)
    end

    tri_mask = mask === nothing ? nothing :
        (reshape(mask, size(mask, 1), 1, size(mask, 2)) .* reshape(mask, 1, size(mask, 1), size(mask, 2)))

    if within_gradient(pairwise_state)
        pairwise_state = pairwise_state .+ m.row_drop(m.tri_mul_out(pairwise_state; mask=tri_mask))
        pairwise_state = pairwise_state .+ m.col_drop(m.tri_mul_in(pairwise_state; mask=tri_mask))
        pairwise_state = pairwise_state .+ m.row_drop(m.tri_att_start(pairwise_state; mask=tri_mask))
        pairwise_state = pairwise_state .+ m.col_drop(m.tri_att_end(pairwise_state; mask=tri_mask))
    else
        pairwise_state .+= m.row_drop(m.tri_mul_out(pairwise_state; mask=tri_mask))
        pairwise_state .+= m.col_drop(m.tri_mul_in(pairwise_state; mask=tri_mask))
        pairwise_state .+= m.row_drop(m.tri_att_start(pairwise_state; mask=tri_mask))
        pairwise_state .+= m.col_drop(m.tri_att_end(pairwise_state; mask=tri_mask))
    end

    pairwise_state = m.mlp_pair(pairwise_state)

    return sequence_state, pairwise_state
end

# ============================================================================
# ResidueMLP: in-place ReLU + in-place residual
# ============================================================================

function (m::ResidueMLP)(x::AnyGPUArray)
    y = m.norm(x)
    y = m.fc1(y)
    if within_gradient(y)
        y = max.(y, 0f0)  # out-of-place ReLU
    else
        y .= max.(y, 0f0)  # in-place ReLU
    end
    y = m.fc2(y)
    y = m.dropout(y)
    if within_gradient(x)
        x = x .+ y  # out-of-place residual
    else
        x .+= y  # in-place residual
    end
    return x
end

# ============================================================================
# StructureModuleTransitionLayer: in-place ReLU + in-place residual
# ============================================================================

function (m::StructureModuleTransitionLayer)(s::AnyGPUArray)
    s0 = s
    s = m.linear_1(s)
    if within_gradient(s)
        s = max.(s, 0f0)
        s = m.linear_2(s)
        s = max.(s, 0f0)
    else
        s .= max.(s, 0f0)
        s = m.linear_2(s)
        s .= max.(s, 0f0)
    end
    s = m.linear_3(s)
    if within_gradient(s)
        s = s .+ s0  # out-of-place residual
    else
        s .+= s0  # in-place residual
    end
    return s
end

# ============================================================================
# AngleResnetBlock: in-place residual
# ============================================================================

function (m::AngleResnetBlock)(a::AnyGPUArray)
    s = a
    a = m.linear_1(max.(a, 0f0))
    a = m.linear_2(max.(a, 0f0))
    if within_gradient(a)
        a = a .+ s  # out-of-place residual
    else
        a .+= s  # in-place residual
    end
    return a
end

# ============================================================================
# SequenceToPair: pre-allocated outer product buffer via similar()
# Avoids separate prod + diff + cat allocations.
# In gradient mode, uses cat() instead of view-writes for AD compatibility.
# ============================================================================

function (m::SequenceToPair)(sequence_state::AnyGPUArray)
    s_norm = m.layernorm(sequence_state)
    s = m.proj(s_norm)
    inner_dim = size(s, 1) ÷ 2
    q = view(s, 1:inner_dim, :, :)
    k = view(s, (inner_dim + 1):(2 * inner_dim), :, :)
    L = size(q, 2)
    B = size(q, 3)
    q_exp = reshape(q, inner_dim, 1, L, B)
    k_exp = reshape(k, inner_dim, L, 1, B)

    if within_gradient(s)
        # Out-of-place: AD-compatible
        prod = q_exp .* k_exp
        diff = q_exp .- k_exp
        x = cat(prod, diff; dims=1)
    else
        # Pre-allocated buffer for product and difference
        x = similar(s, 2 * inner_dim, L, L, B)
        prod_view = @view x[1:inner_dim, :, :, :]
        diff_view = @view x[inner_dim+1:end, :, :, :]
        @. prod_view = q_exp * k_exp
        @. diff_view = q_exp - k_exp
    end

    x = m.o_proj(x)
    return x
end
