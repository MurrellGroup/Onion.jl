using Flux
using NNlib

@concrete struct AttentionPairBias <: Layer
    proj_q
    proj_k
    proj_v
    proj_g
    proj_o
    proj_z_norm
    proj_z
    q_norm
    k_norm
    c_s::Int
    num_heads::Int
    head_dim::Int
    inf
    compute_pair_bias::Bool
    use_qk_norm::Bool
end

@layer AttentionPairBias

function AttentionPairBias(
    c_s::Int,
    c_z::Int,
    num_heads::Int;
    inf::Real=1f6,
    compute_pair_bias::Bool=true,
    use_qk_norm::Bool=false,
)
    @assert c_s % num_heads == 0 "c_s must be divisible by num_heads"
    head_dim = c_s ÷ num_heads

    proj_q = Dense(c_s => c_s)
    proj_k = Dense(c_s => c_s, bias=false)
    proj_v = Dense(c_s => c_s, bias=false)
    proj_g = Dense(c_s => c_s, bias=false)

    torch_linear_init!(proj_q.weight, proj_q.bias)
    torch_linear_init!(proj_k.weight)
    torch_linear_init!(proj_v.weight)
    torch_linear_init!(proj_g.weight)

    q_norm = use_qk_norm ? BGLayerNorm(head_dim; eps=1f-5) : identity
    k_norm = use_qk_norm ? BGLayerNorm(head_dim; eps=1f-5) : identity

    if compute_pair_bias
        proj_z_norm = BGLayerNorm(c_z; eps=1f-5)
        proj_z = Dense(c_z => num_heads, bias=false)
        torch_linear_init!(proj_z.weight)
    else
        proj_z_norm = identity
        proj_z = identity
    end

    proj_o = Dense(c_s => c_s, bias=false)
    final_init!(proj_o.weight)

    return AttentionPairBias(
        proj_q,
        proj_k,
        proj_v,
        proj_g,
        proj_o,
        proj_z_norm,
        proj_z,
        q_norm,
        k_norm,
        c_s,
        num_heads,
        head_dim,
        inf,
        compute_pair_bias,
        use_qk_norm,
    )
end

function _bg_scaled_dot_product_attention(q, k, v, mask_bias, bias)
    T = eltype(q)
    @assert eltype(k) === T && eltype(v) === T && eltype(bias) === T && eltype(mask_bias) === T "attention input eltypes must match: got q=$(eltype(q)), k=$(eltype(k)), v=$(eltype(v)), bias=$(eltype(bias)), mask=$(eltype(mask_bias))"

    # Combine bias (H, Q, K, B) + mask_bias (1, 1, K, B), then convert to
    # flash attention layout (K, Q, H, B) for the dispatch hook.
    combined_bias = bias .+ mask_bias
    flash_bias = permutedims(combined_bias, (3, 2, 1, 4))

    # Dispatch to best available backend: CPU → ONIONop (KA) → OnionTile (cuTile)
    # OnionTile handles non-pow2 head dims via transparent padding.
    # Default scale = 1/sqrt(D) matches the manual scale used previously.
    return flash_attention_bias_forward(q, k, v, flash_bias)
end

function (layer::AttentionPairBias)(s, z, mask, k_in; multiplicity::Int=1)
    c_s = layer.c_s
    n_q = size(s, 2)
    n_k = size(k_in, 2)
    b = size(s, 3)

    q = layer.proj_q(s)
    k = layer.proj_k(k_in)
    v = layer.proj_v(k_in)

    q = reshape(q, layer.head_dim, layer.num_heads, n_q, b)
    k = reshape(k, layer.head_dim, layer.num_heads, n_k, b)
    v = reshape(v, layer.head_dim, layer.num_heads, n_k, b)

    q = permutedims(q, (1, 3, 2, 4)) # (d, n, h, b)
    k = permutedims(k, (1, 3, 2, 4))
    v = permutedims(v, (1, 3, 2, 4))

    if layer.use_qk_norm
        q = layer.q_norm(q)
        k = layer.k_norm(k)
    end

    if layer.compute_pair_bias
        bias = layer.proj_z(layer.proj_z_norm(z)) # (H, N, N, B)
    else
        bias = z
    end

    if multiplicity != 1
        bias = repeat(bias, 1, 1, 1, multiplicity)
    end

    g = layer.proj_g(s)
    g = NNlib.sigmoid.(g)

    # mask is (N_k, B)
    mask_bias = (1 .- mask) .* (-layer.inf)
    mask_bias = reshape(mask_bias, 1, 1, size(mask_bias, 1), size(mask_bias, 2))

    o = _bg_scaled_dot_product_attention(q, k, v, mask_bias, bias)
    o = permutedims(o, (1, 3, 2, 4))
    o = reshape(o, c_s, n_q, b)

    o = o .* g
    o = layer.proj_o(o)

    return o
end
