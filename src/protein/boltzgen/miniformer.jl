using Flux

@concrete struct MiniformerModule <: Layer
    token_z::Int
    use_s_to_z::Bool
    num_blocks::Int
    dropout
    num_heads::Int
    post_layer_norm::Bool
    layers
    activation_checkpointing::Bool
end

@layer MiniformerModule

function MiniformerModule(
    token_s::Int,
    token_z::Int,
    num_blocks::Int;
    num_heads::Int=16,
    dropout::Real=0.25,
    use_s_to_z::Bool=false,
    post_layer_norm::Bool=false,
    activation_checkpointing::Bool=false,
)
    layers = [
        MiniformerLayer(token_s, token_z, num_heads; dropout=dropout, post_layer_norm=post_layer_norm, use_s_to_z=use_s_to_z)
        for _ in 1:num_blocks
    ]
    return MiniformerModule(
        token_z,
        use_s_to_z,
        num_blocks,
        dropout,
        num_heads,
        post_layer_norm,
        layers,
        activation_checkpointing,
    )
end

function (m::MiniformerModule)(s, z, mask, pair_mask; use_kernels::Bool=false)
    for layer in m.layers
        s, z = layer(s, z, mask, pair_mask; use_kernels=use_kernels)
    end
    return s, z
end

@concrete struct MiniformerLayer <: Layer
    token_z::Int
    dropout
    num_heads::Int
    post_layer_norm::Bool
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

@layer MiniformerLayer

function MiniformerLayer(
    token_s::Int,
    token_z::Int,
    num_heads::Int=16;
    dropout::Real=0.25,
    post_layer_norm::Bool=false,
    use_s_to_z::Bool=false,
)
    pre_norm_s = BGLayerNorm(token_s; eps=1f-5)
    attention = AttentionPairBias(token_s, token_z, num_heads)
    triangular = MiniTriangularUpdate(token_z)
    transition_s = Transition(token_s, token_s * 4)
    transition_z = Transition(token_z, token_z * 4)

    s_post_norm = post_layer_norm ? BGLayerNorm(token_s; eps=1f-5) : identity

    s_to_z_1 = use_s_to_z ? Dense(token_s => token_z, bias=false) : identity
    s_to_z_2 = use_s_to_z ? Dense(token_s => token_z, bias=false) : identity
    if use_s_to_z
        torch_linear_init!(s_to_z_1.weight)
        torch_linear_init!(s_to_z_2.weight)
    end

    return MiniformerLayer(
        token_z,
        dropout,
        num_heads,
        post_layer_norm,
        use_s_to_z,
        pre_norm_s,
        attention,
        triangular,
        transition_s,
        transition_z,
        s_post_norm,
        s_to_z_1,
        s_to_z_2,
    )
end

function (layer::MiniformerLayer)(s, z, mask, pair_mask; use_kernels::Bool=false)
    if layer.use_s_to_z
        s1 = layer.s_to_z_1(s)
        s2 = layer.s_to_z_2(s)
        z = z .+ reshape(s1, size(s1,1), size(s1,2), 1, size(s1,3)) .+
            reshape(s2, size(s2,1), 1, size(s2,2), size(s2,3))
    end

    dropout = get_dropout_mask_rowise(layer.dropout, z, bg_istraining())
    z = z .+ dropout .* layer.triangular(z; mask=pair_mask, use_kernels=use_kernels)
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

@concrete struct MiniformerNoSeqModule <: Layer
    token_z::Int
    num_blocks::Int
    dropout
    post_layer_norm::Bool
    layers
    activation_checkpointing::Bool
end

@layer MiniformerNoSeqModule

function MiniformerNoSeqModule(
    token_z::Int,
    num_blocks::Int;
    dropout::Real=0.25,
    post_layer_norm::Bool=false,
    activation_checkpointing::Bool=false,
)
    layers = [
        MiniformerNoSeqLayer(token_z; dropout=dropout, post_layer_norm=post_layer_norm)
        for _ in 1:num_blocks
    ]
    return MiniformerNoSeqModule(
        token_z,
        num_blocks,
        dropout,
        post_layer_norm,
        layers,
        activation_checkpointing,
    )
end

function (m::MiniformerNoSeqModule)(z, pair_mask; use_kernels::Bool=false)
    for layer in m.layers
        z = layer(z, pair_mask; use_kernels=use_kernels)
    end
    return z
end

@concrete struct MiniformerNoSeqLayer <: Layer
    token_z::Int
    dropout
    post_layer_norm::Bool
    triangular
    transition_z
    z_post_norm
end

@layer MiniformerNoSeqLayer

function MiniformerNoSeqLayer(
    token_z::Int;
    dropout::Real=0.25,
    post_layer_norm::Bool=false,
)
    triangular = MiniTriangularUpdate(token_z)
    transition_z = Transition(token_z, token_z * 4)
    z_post_norm = post_layer_norm ? BGLayerNorm(token_z; eps=1f-5) : identity

    return MiniformerNoSeqLayer(
        token_z,
        dropout,
        post_layer_norm,
        triangular,
        transition_z,
        z_post_norm,
    )
end

function (layer::MiniformerNoSeqLayer)(z, pair_mask; chunk_size_tri_attn::Union{Nothing,Int}=nothing, use_kernels::Bool=false)
    dropout = get_dropout_mask_rowise(layer.dropout, z, bg_istraining())
    z = z .+ dropout .* layer.triangular(z; mask=pair_mask, use_kernels=use_kernels)
    z = z .+ layer.transition_z(z)
    z = layer.z_post_norm(z)
    return z
end
