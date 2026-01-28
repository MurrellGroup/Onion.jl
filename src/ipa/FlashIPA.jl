@concrete struct FlashIPA <: Layer
    qkv_proj
    qk_point_proj
    v_point_proj
    o_proj
    b_pair_proj
    head_weights
    num_heads
    head_dim
    num_query_points
    num_point_values
end

function FlashIPA(;
    hidden_size::Int, num_heads::Int,
    head_dim::Int = hidden_size ÷ num_heads,
    num_query_points::Int, num_point_values::Int,
    pair_dim::Int,
)
    return FlashIPA(
        Linear(hidden_size => head_dim * num_heads * 3),
        Linear(hidden_size => 3 * num_heads * num_query_points * 2),
        Linear(hidden_size => 3 * num_heads * num_point_values),
        Linear(num_heads * (head_dim + num_point_values * (3 + 1)) => hidden_size),
        Linear(pair_dim => num_heads),
        log.(exp.(ones(Float32, num_heads)) .- 1), # 1 after softplus[]
        num_heads, head_dim, num_query_points, num_point_values
    )
end

norm²(x; dims) = sum(abs2, x; dims)

"""
- `s`: (c h) l b
- `R`: 3 3 l b
- `t`: 3 1 l b
- `z`: dz ql kl b
"""
function (layer::FlashIPA)(s, z, (R, t); kws...)
    d, h, c, pₖ, pᵥ = 3, layer.num_heads, layer.head_dim, layer.num_query_points, layer.num_point_values
    q, k, v = split_axis(layer.qkv_proj(s), 3, dims=1)
    q, k, v = rearrange.((q, k, v), einops"(c h) l b -> c l h b"; h)
    qᵖ, kᵖ = split_axis(layer.qk_point_proj(s), 2, dims=1)
    vᵖ = layer.v_point_proj(s)
    qᵖ, kᵖ, vᵖ = rearrange.((qᵖ, kᵖ, vᵖ), einops"(d h p) l b -> d p l h b"; d, h)
    wC, wL = √(2 / 9pₖ), 1 / √3
    wC, wL, cf = map(x -> ofeltype(x, q), (wC, wL, c))
    t = rearrange(t, einops"d2 d1 l b -> d2 d1 l 1 b")
    qᵖ′ = t .+ einsum(R, qᵖ, einops"d2 d1 l b, d1 p l h b -> d2 p l h b")
    # c + (3+1+1)pₖ
    q̂ = [
        q
        rearrange(qᵖ′, einops"d p l h b -> (d p) l h b"; d)
        reduce(norm², qᵖ′, einops"d p l h b -> p l h b"; d)
        ones_like(qᵖ′, size(qᵖ′)[2:end])
    ]
    γ = softplus.(layer.head_weights)
    coef = reshape(γ .* (wL * wC), einops"h -> 1 1 h")
    coef_neg_half = -0.5f0 .* coef
    kᵖ′ = t .+ einsum(R, kᵖ, einops"d2 d1 l b, d1 p l h b -> d2 p l h b")
    # c + (3+1+1)pₖ
    k̂ = [
        (wL / √cf) .* k
        coef .* rearrange(kᵖ′, einops"d p l h b -> (d p) l h b")
        repeat(coef_neg_half, einops"1 1 h -> p l h b"; parse_shape(kᵖ, einops"_ p l h b")...)
        coef_neg_half .* reduce(norm², kᵖ′, einops"d p l h b -> p l h b"; d)
    ]
    # c + 3pᵥ
    vᵖ′ = t .+ einsum(R, vᵖ, einops"d2 d1 l b, d1 p l h b -> d2 p l h b")
    v̂ = [
        v
        rearrange(vᵖ′, einops"d p l h b -> (d p) l h b")
    ]
    ô = Ops.attention(q̂, k̂, v̂; pair=layer.b_pair_proj(z), kws...)
    o, oᵖ = copy(selectdim(ô, 1, 1:c)), copy(selectdim(ô, 1, c+1:size(ô, 1)))
    # c + (3+1)pᵥ
    O = rearrange([
        o
        oᵖ
        reduce(norm², oᵖ, einops"(d p) l h b -> p l h b"; d)
    ], einops"c l h b -> (c h) l b")
    s̃ = layer.o_proj(O)
    return s̃
end
