using Flux

const BG_CHUNK_SIZE_THRESHOLD = 512

@concrete struct PairformerModule <: Layer
    token_z::Int
    num_blocks::Int
    dropout
    num_heads::Int
    post_layer_norm::Bool
    activation_checkpointing::Bool
    layers
end

@layer PairformerModule

function PairformerModule(
    token_s::Int,
    token_z::Int,
    num_blocks::Int;
    num_heads::Int=16,
    dropout::Real=0.25,
    pairwise_head_width::Int=32,
    pairwise_num_heads::Int=4,
    post_layer_norm::Bool=false,
    activation_checkpointing::Bool=false,
)
    layers = [
        PairformerLayer(
            token_s,
            token_z,
            num_heads;
            dropout=dropout,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
            post_layer_norm=post_layer_norm,
        )
        for _ in 1:num_blocks
    ]

    return PairformerModule(
        token_z,
        num_blocks,
        dropout,
        num_heads,
        post_layer_norm,
        activation_checkpointing,
        layers,
    )
end

function (m::PairformerModule)(s, z, mask, pair_mask; use_kernels::Bool=false)
    chunk_size_tri_attn = if !bg_istraining()
        size(z, 2) > BG_CHUNK_SIZE_THRESHOLD ? 128 : 512
    else
        nothing
    end

    for layer in m.layers
        s, z = layer(s, z, mask, pair_mask; chunk_size_tri_attn=chunk_size_tri_attn, use_kernels=use_kernels)
    end
    return s, z
end

@concrete struct PairformerLayer <: Layer
    token_z::Int
    dropout
    num_heads::Int
    post_layer_norm::Bool
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

@layer PairformerLayer

function PairformerLayer(
    token_s::Int,
    token_z::Int,
    num_heads::Int=16;
    dropout::Real=0.25,
    pairwise_head_width::Int=32,
    pairwise_num_heads::Int=4,
    post_layer_norm::Bool=false,
)
    pre_norm_s = BGLayerNorm(token_s; eps=1f-5)
    attention = AttentionPairBias(token_s, token_z, num_heads)

    tri_mul_out = BGTriangleMultiplicationOutgoing(token_z)
    tri_mul_in = BGTriangleMultiplicationIncoming(token_z)

    tri_att_start = TriangleAttentionStartingNode(token_z, pairwise_head_width, pairwise_num_heads; inf=1f9)
    tri_att_end = TriangleAttentionEndingNode(token_z, pairwise_head_width, pairwise_num_heads; inf=1f9)

    transition_s = Transition(token_s, token_s * 4)
    transition_z = Transition(token_z, token_z * 4)

    s_post_norm = post_layer_norm ? BGLayerNorm(token_s; eps=1f-5) : identity

    return PairformerLayer(
        token_z,
        dropout,
        num_heads,
        post_layer_norm,
        pre_norm_s,
        attention,
        tri_mul_out,
        tri_mul_in,
        tri_att_start,
        tri_att_end,
        transition_s,
        transition_z,
        s_post_norm,
    )
end

function (layer::PairformerLayer)(
    s,
    z,
    mask,
    pair_mask;
    chunk_size_tri_attn::Union{Nothing,Int}=nothing,
    use_kernels::Bool=false,
    use_cuequiv_mul::Bool=false,
    use_cuequiv_attn::Bool=false,
)
    dropout = get_dropout_mask(layer.dropout, z, bg_istraining())
    z = z .+ dropout .* layer.tri_mul_out(z; mask=pair_mask, use_kernels=use_cuequiv_mul || use_kernels)

    dropout = get_dropout_mask(layer.dropout, z, bg_istraining())
    z = z .+ dropout .* layer.tri_mul_in(z; mask=pair_mask, use_kernels=use_cuequiv_mul || use_kernels)

    dropout = get_dropout_mask(layer.dropout, z, bg_istraining())
    z = z .+ dropout .* layer.tri_att_start(z; mask=pair_mask, chunk_size=chunk_size_tri_attn)

    dropout = get_dropout_mask(layer.dropout, z, bg_istraining(); columnwise=true)
    z = z .+ dropout .* layer.tri_att_end(z; mask=pair_mask, chunk_size=chunk_size_tri_attn)

    z = z .+ layer.transition_z(z)

    s_f = Float32.(s)
    z_f = Float32.(z)
    mask_f = Float32.(mask)

    s_normed = layer.pre_norm_s(s_f)
    s = s_f .+ layer.attention(s_normed, z_f, mask_f, s_normed)
    s = s .+ layer.transition_s(s)
    s = layer.s_post_norm(s)

    return s, z
end

@concrete struct PairformerNoSeqModule <: Layer
    token_z::Int
    num_blocks::Int
    dropout
    post_layer_norm::Bool
    activation_checkpointing::Bool
    layers
end

@layer PairformerNoSeqModule

function PairformerNoSeqModule(
    token_z::Int,
    num_blocks::Int;
    dropout::Real=0.25,
    pairwise_head_width::Int=32,
    pairwise_num_heads::Int=4,
    post_layer_norm::Bool=false,
    activation_checkpointing::Bool=false,
)
    layers = [
        PairformerNoSeqLayer(
            token_z;
            dropout=dropout,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
            post_layer_norm=post_layer_norm,
        )
        for _ in 1:num_blocks
    ]

    return PairformerNoSeqModule(
        token_z,
        num_blocks,
        dropout,
        post_layer_norm,
        activation_checkpointing,
        layers,
    )
end

function (m::PairformerNoSeqModule)(z, pair_mask; use_kernels::Bool=false)
    chunk_size_tri_attn = if !bg_istraining()
        size(z, 2) > BG_CHUNK_SIZE_THRESHOLD ? 128 : 512
    else
        nothing
    end

    for layer in m.layers
        z = layer(z, pair_mask; chunk_size_tri_attn=chunk_size_tri_attn, use_kernels=use_kernels)
    end
    return z
end

@concrete struct PairformerNoSeqLayer <: Layer
    token_z::Int
    dropout
    post_layer_norm::Bool
    tri_mul_out
    tri_mul_in
    tri_att_start
    tri_att_end
    transition_z
end

@layer PairformerNoSeqLayer

function PairformerNoSeqLayer(
    token_z::Int;
    dropout::Real=0.25,
    pairwise_head_width::Int=32,
    pairwise_num_heads::Int=4,
    post_layer_norm::Bool=false,
)
    tri_mul_out = BGTriangleMultiplicationOutgoing(token_z)
    tri_mul_in = BGTriangleMultiplicationIncoming(token_z)

    tri_att_start = TriangleAttentionStartingNode(token_z, pairwise_head_width, pairwise_num_heads; inf=1f9)
    tri_att_end = TriangleAttentionEndingNode(token_z, pairwise_head_width, pairwise_num_heads; inf=1f9)

    transition_z = Transition(token_z, token_z * 4)

    return PairformerNoSeqLayer(
        token_z,
        dropout,
        post_layer_norm,
        tri_mul_out,
        tri_mul_in,
        tri_att_start,
        tri_att_end,
        transition_z,
    )
end

function (layer::PairformerNoSeqLayer)(
    z,
    pair_mask;
    chunk_size_tri_attn::Union{Nothing,Int}=nothing,
    use_kernels::Bool=false,
    use_cuequiv_mul::Bool=false,
    use_cuequiv_attn::Bool=false,
)
    dropout = get_dropout_mask(layer.dropout, z, bg_istraining())
    z = z .+ dropout .* layer.tri_mul_out(z; mask=pair_mask, use_kernels=use_cuequiv_mul || use_kernels)

    dropout = get_dropout_mask(layer.dropout, z, bg_istraining())
    z = z .+ dropout .* layer.tri_mul_in(z; mask=pair_mask, use_kernels=use_cuequiv_mul || use_kernels)

    dropout = get_dropout_mask(layer.dropout, z, bg_istraining())
    z = z .+ dropout .* layer.tri_att_start(z; mask=pair_mask, chunk_size=chunk_size_tri_attn)

    dropout = get_dropout_mask(layer.dropout, z, bg_istraining(); columnwise=true)
    z = z .+ dropout .* layer.tri_att_end(z; mask=pair_mask, chunk_size=chunk_size_tri_attn)

    z = z .+ layer.transition_z(z)
    return z
end
