# AnyGPUArray layer overrides for performance-critical protein layers.
# Uses in-place operations (.+=, .=) that work on any GPU backend.
# No buffer pool, no CUDA.unsafe_free! — those are OnionTile-specific.

using GPUArraysCore: AnyGPUArray

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

    # In-place residual additions
    sequence_state .+= m.drop(y)
    sequence_state = m.mlp_seq(sequence_state)

    pairwise_state .+= m.sequence_to_pair(sequence_state)

    tri_mask = mask === nothing ? nothing :
        (reshape(mask, size(mask, 1), 1, size(mask, 2)) .* reshape(mask, 1, size(mask, 1), size(mask, 2)))
    pairwise_state .+= m.row_drop(m.tri_mul_out(pairwise_state; mask=tri_mask))
    pairwise_state .+= m.col_drop(m.tri_mul_in(pairwise_state; mask=tri_mask))
    pairwise_state .+= m.row_drop(m.tri_att_start(pairwise_state; mask=tri_mask))
    pairwise_state .+= m.col_drop(m.tri_att_end(pairwise_state; mask=tri_mask))

    pairwise_state = m.mlp_pair(pairwise_state)

    return sequence_state, pairwise_state
end

# ============================================================================
# ResidueMLP: in-place ReLU + in-place residual
# ============================================================================

function (m::ResidueMLP)(x::AnyGPUArray)
    y = m.norm(x)
    y = m.fc1(y)
    y .= max.(y, 0f0)  # in-place ReLU
    y = m.fc2(y)
    y = m.dropout(y)
    x .+= y  # in-place residual
    return x
end

# ============================================================================
# StructureModuleTransitionLayer: in-place ReLU + in-place residual
# ============================================================================

function (m::StructureModuleTransitionLayer)(s::AnyGPUArray)
    s0 = s
    s = m.linear_1(s)
    s .= max.(s, 0f0)  # in-place ReLU
    s = m.linear_2(s)
    s .= max.(s, 0f0)  # in-place ReLU
    s = m.linear_3(s)
    s .+= s0  # in-place residual
    return s
end

# ============================================================================
# AngleResnetBlock: in-place residual
# ============================================================================

function (m::AngleResnetBlock)(a::AnyGPUArray)
    s = a
    a = m.linear_1(max.(a, 0f0))
    a = m.linear_2(max.(a, 0f0))
    a .+= s  # in-place residual
    return a
end

# ============================================================================
# SequenceToPair: pre-allocated outer product buffer via similar()
# Avoids separate prod + diff + cat allocations.
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

    # Pre-allocated buffer for product and difference
    x = similar(s, 2 * inner_dim, L, L, B)
    prod_view = @view x[1:inner_dim, :, :, :]
    diff_view = @view x[inner_dim+1:end, :, :, :]
    @. prod_view = q_exp * k_exp
    @. diff_view = q_exp - k_exp

    x = m.o_proj(x)
    return x
end
