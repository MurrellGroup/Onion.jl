# ──── ESMFold TriangularSelfAttentionBlock ────

"""
    TriangularSelfAttentionBlock(seq_dim, pair_dim, seq_head_width, pair_head_width; dropout=0)

The main ESMFold evoformer block. Alternates sequence attention (with pair bias)
and pairwise triangle operations.

Forward: `(sequence_state, pairwise_state; mask=nothing)` → `(sequence_state, pairwise_state)`
"""
@concrete struct TriangularSelfAttentionBlock <: Layer
    layernorm_1
    sequence_to_pair
    pair_to_sequence
    seq_attention
    tri_mul_out
    tri_mul_in
    tri_att_start
    tri_att_end
    mlp_seq
    mlp_pair
    drop
    row_drop
    col_drop
end

function TriangularSelfAttentionBlock(
    sequence_state_dim::Int,
    pairwise_state_dim::Int,
    sequence_head_width::Int,
    pairwise_head_width::Int;
    dropout::Real=0,
)
    sequence_num_heads = sequence_state_dim ÷ sequence_head_width
    pairwise_num_heads = pairwise_state_dim ÷ pairwise_head_width

    layernorm_1       = LayerNorm(sequence_state_dim)
    sequence_to_pair  = SequenceToPair(sequence_state_dim, pairwise_state_dim ÷ 2, pairwise_state_dim)
    pair_to_sequence  = PairToSequence(pairwise_state_dim, sequence_num_heads)

    seq_attention = ESMFoldAttention(sequence_state_dim, sequence_num_heads, sequence_head_width; gated=true)

    tri_mul_out   = TriangleMultiplicationOutgoing(pairwise_state_dim, pairwise_state_dim)
    tri_mul_in    = TriangleMultiplicationIncoming(pairwise_state_dim, pairwise_state_dim)
    tri_att_start = TriangleAttentionStartingNode(pairwise_state_dim, pairwise_head_width, pairwise_num_heads)
    tri_att_end   = TriangleAttentionEndingNode(pairwise_state_dim, pairwise_head_width, pairwise_num_heads)

    mlp_seq  = ResidueMLP(sequence_state_dim, 4 * sequence_state_dim; dropout=dropout)
    mlp_pair = ResidueMLP(pairwise_state_dim, 4 * pairwise_state_dim; dropout=dropout)

    drop     = SharedDropout(dropout, [3])
    row_drop = SharedDropout(dropout * 2, [2])
    col_drop = SharedDropout(dropout * 2, [1])

    return TriangularSelfAttentionBlock(
        layernorm_1, sequence_to_pair, pair_to_sequence, seq_attention,
        tri_mul_out, tri_mul_in, tri_att_start, tri_att_end,
        mlp_seq, mlp_pair, drop, row_drop, col_drop,
    )
end

function (m::TriangularSelfAttentionBlock)(sequence_state, pairwise_state; mask=nothing)
    bias = m.pair_to_sequence(pairwise_state)
    y = m.layernorm_1(sequence_state)
    y, _ = m.seq_attention(y; mask=mask, bias=bias)

    sequence_state = sequence_state .+ m.drop(y)
    sequence_state = m.mlp_seq(sequence_state)

    pairwise_state = pairwise_state .+ m.sequence_to_pair(sequence_state)

    tri_mask = if mask === nothing
        nothing
    else
        reshape(mask, size(mask, 1), 1, size(mask, 2)) .*
        reshape(mask, 1, size(mask, 1), size(mask, 2))
    end

    pairwise_state = pairwise_state .+ m.row_drop(m.tri_mul_out(pairwise_state; mask=tri_mask))
    pairwise_state = pairwise_state .+ m.col_drop(m.tri_mul_in(pairwise_state; mask=tri_mask))
    pairwise_state = pairwise_state .+ m.row_drop(m.tri_att_start(pairwise_state; mask=tri_mask))
    pairwise_state = pairwise_state .+ m.col_drop(m.tri_att_end(pairwise_state; mask=tri_mask))

    pairwise_state = m.mlp_pair(pairwise_state)

    return sequence_state, pairwise_state
end
