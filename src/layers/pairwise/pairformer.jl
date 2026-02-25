# ──── PairformerLayer / PairformerModule ────

"""
    PairformerLayer(token_s, token_z; num_heads=16, dropout=0.25, ...)

Single BoltzGen-style Pairformer block: pair track (4 triangle ops + transition)
followed by sequence track (pair-biased attention + transition).
"""
@concrete struct PairformerLayer <: Layer
    dropout
    pre_norm_s
    attention
    tri_mul_out
    tri_mul_in
    tri_att_start
    tri_att_end
    transition_s
    transition_z
    s_post_norm
end

function PairformerLayer(
    token_s::Int,
    token_z::Int;
    num_heads::Int=16,
    dropout::Real=0.25,
    pairwise_head_width::Int=32,
    pairwise_num_heads::Int=4,
    post_layer_norm::Bool=false,
)
    pre_norm_s = LayerNorm(token_s)
    attention  = AttentionPairBias(token_s, token_z, num_heads)

    tri_mul_out  = BGTriangleMultiplicationOutgoing(token_z)
    tri_mul_in   = BGTriangleMultiplicationIncoming(token_z)
    tri_att_start = TriangleAttentionStartingNode(token_z, pairwise_head_width, pairwise_num_heads)
    tri_att_end   = TriangleAttentionEndingNode(token_z, pairwise_head_width, pairwise_num_heads)

    transition_s = Transition(token_s, token_s * 4)
    transition_z = Transition(token_z, token_z * 4)

    s_post_norm = post_layer_norm ? LayerNorm(token_s) : identity

    return PairformerLayer(
        dropout, pre_norm_s, attention,
        tri_mul_out, tri_mul_in, tri_att_start, tri_att_end,
        transition_s, transition_z, s_post_norm,
    )
end

function (layer::PairformerLayer)(s, z, mask, pair_mask)
    z = z .+ get_dropout_mask(layer.dropout, z) .* layer.tri_mul_out(z; mask=pair_mask)
    z = z .+ get_dropout_mask(layer.dropout, z) .* layer.tri_mul_in(z; mask=pair_mask)
    z = z .+ get_dropout_mask(layer.dropout, z) .* layer.tri_att_start(z; mask=pair_mask)
    z = z .+ get_dropout_mask(layer.dropout, z; columnwise=true) .* layer.tri_att_end(z; mask=pair_mask)
    z = z .+ layer.transition_z(z)

    s_normed = layer.pre_norm_s(s)
    s = s .+ layer.attention(s_normed, z, mask, s_normed)
    s = s .+ layer.transition_s(s)
    s = layer.s_post_norm(s)

    return s, z
end

"""
    PairformerModule(token_s, token_z, num_blocks; ...)

Stack of `PairformerLayer`s.
"""
@concrete struct PairformerModule <: Layer
    layers
end

function PairformerModule(
    token_s::Int,
    token_z::Int,
    num_blocks::Int;
    kws...
)
    layers = [PairformerLayer(token_s, token_z; kws...) for _ in 1:num_blocks]
    return PairformerModule(layers)
end

function (m::PairformerModule)(s, z, mask, pair_mask)
    for layer in m.layers
        s, z = layer(s, z, mask, pair_mask)
    end
    return s, z
end

# ──── PairformerNoSeqLayer / PairformerNoSeqModule ────

"""
    PairformerNoSeqLayer(token_z; dropout=0.25, ...)

Pair-only Pairformer block (no sequence track).
"""
@concrete struct PairformerNoSeqLayer <: Layer
    dropout
    tri_mul_out
    tri_mul_in
    tri_att_start
    tri_att_end
    transition_z
end

function PairformerNoSeqLayer(
    token_z::Int;
    dropout::Real=0.25,
    pairwise_head_width::Int=32,
    pairwise_num_heads::Int=4,
)
    tri_mul_out   = BGTriangleMultiplicationOutgoing(token_z)
    tri_mul_in    = BGTriangleMultiplicationIncoming(token_z)
    tri_att_start = TriangleAttentionStartingNode(token_z, pairwise_head_width, pairwise_num_heads)
    tri_att_end   = TriangleAttentionEndingNode(token_z, pairwise_head_width, pairwise_num_heads)
    transition_z  = Transition(token_z, token_z * 4)

    return PairformerNoSeqLayer(
        dropout, tri_mul_out, tri_mul_in, tri_att_start, tri_att_end, transition_z,
    )
end

function (layer::PairformerNoSeqLayer)(z, pair_mask)
    z = z .+ get_dropout_mask(layer.dropout, z) .* layer.tri_mul_out(z; mask=pair_mask)
    z = z .+ get_dropout_mask(layer.dropout, z) .* layer.tri_mul_in(z; mask=pair_mask)
    z = z .+ get_dropout_mask(layer.dropout, z) .* layer.tri_att_start(z; mask=pair_mask)
    z = z .+ get_dropout_mask(layer.dropout, z; columnwise=true) .* layer.tri_att_end(z; mask=pair_mask)
    z = z .+ layer.transition_z(z)
    return z
end

@concrete struct PairformerNoSeqModule <: Layer
    layers
end

function PairformerNoSeqModule(
    token_z::Int,
    num_blocks::Int;
    kws...
)
    layers = [PairformerNoSeqLayer(token_z; kws...) for _ in 1:num_blocks]
    return PairformerNoSeqModule(layers)
end

function (m::PairformerNoSeqModule)(z, pair_mask)
    for layer in m.layers
        z = layer(z, pair_mask)
    end
    return z
end

# ──── MiniformerLayer / MiniformerModule ────

"""
    MiniformerLayer(token_s, token_z; num_heads=16, dropout=0.25, ...)

Single Miniformer block: uses `MiniTriangularUpdate` (compact, does both outgoing+incoming)
instead of 4 separate triangle ops.
"""
@concrete struct MiniformerLayer <: Layer
    dropout
    use_s_to_z::Bool
    pre_norm_s
    attention
    triangular
    transition_s
    transition_z
    s_post_norm
    s_to_z_1
    s_to_z_2
end

function MiniformerLayer(
    token_s::Int,
    token_z::Int;
    num_heads::Int=16,
    dropout::Real=0.25,
    post_layer_norm::Bool=false,
    use_s_to_z::Bool=false,
)
    pre_norm_s   = LayerNorm(token_s)
    attention    = AttentionPairBias(token_s, token_z, num_heads)
    triangular   = MiniTriangularUpdate(token_z)
    transition_s = Transition(token_s, token_s * 4)
    transition_z = Transition(token_z, token_z * 4)

    s_post_norm = post_layer_norm ? LayerNorm(token_s) : identity

    s_to_z_1 = use_s_to_z ? Linear(token_s => token_z, bias=false) : identity
    s_to_z_2 = use_s_to_z ? Linear(token_s => token_z, bias=false) : identity

    return MiniformerLayer(
        dropout, use_s_to_z, pre_norm_s, attention, triangular,
        transition_s, transition_z, s_post_norm, s_to_z_1, s_to_z_2,
    )
end

function (layer::MiniformerLayer)(s, z, mask, pair_mask)
    if layer.use_s_to_z
        s1 = layer.s_to_z_1(s)
        s2 = layer.s_to_z_2(s)
        z = z .+ reshape(s1, size(s1, 1), size(s1, 2), 1, size(s1, 3)) .+
            reshape(s2, size(s2, 1), 1, size(s2, 2), size(s2, 3))
    end

    z = z .+ get_dropout_mask_rowise(layer.dropout, z) .* layer.triangular(z; mask=pair_mask)
    z = z .+ layer.transition_z(z)

    s_normed = layer.pre_norm_s(s)
    s = s .+ layer.attention(s_normed, z, mask, s_normed)
    s = s .+ layer.transition_s(s)
    s = layer.s_post_norm(s)

    return s, z
end

@concrete struct MiniformerModule <: Layer
    layers
end

function MiniformerModule(
    token_s::Int,
    token_z::Int,
    num_blocks::Int;
    kws...
)
    layers = [MiniformerLayer(token_s, token_z; kws...) for _ in 1:num_blocks]
    return MiniformerModule(layers)
end

function (m::MiniformerModule)(s, z, mask, pair_mask)
    for layer in m.layers
        s, z = layer(s, z, mask, pair_mask)
    end
    return s, z
end

# ──── MiniformerNoSeqLayer / MiniformerNoSeqModule ────

"""
    MiniformerNoSeqLayer(token_z; dropout=0.25, ...)

Pair-only Miniformer block.
"""
@concrete struct MiniformerNoSeqLayer <: Layer
    dropout
    triangular
    transition_z
    z_post_norm
end

function MiniformerNoSeqLayer(
    token_z::Int;
    dropout::Real=0.25,
    post_layer_norm::Bool=false,
)
    triangular   = MiniTriangularUpdate(token_z)
    transition_z = Transition(token_z, token_z * 4)
    z_post_norm  = post_layer_norm ? LayerNorm(token_z) : identity

    return MiniformerNoSeqLayer(dropout, triangular, transition_z, z_post_norm)
end

function (layer::MiniformerNoSeqLayer)(z, pair_mask)
    z = z .+ get_dropout_mask_rowise(layer.dropout, z) .* layer.triangular(z; mask=pair_mask)
    z = z .+ layer.transition_z(z)
    z = layer.z_post_norm(z)
    return z
end

@concrete struct MiniformerNoSeqModule <: Layer
    layers
end

function MiniformerNoSeqModule(
    token_z::Int,
    num_blocks::Int;
    kws...
)
    layers = [MiniformerNoSeqLayer(token_z; kws...) for _ in 1:num_blocks]
    return MiniformerNoSeqModule(layers)
end

function (m::MiniformerNoSeqModule)(z, pair_mask)
    for layer in m.layers
        z = layer(z, pair_mask)
    end
    return z
end
