include("utils.jl")

@concrete struct InvariantPointAttention <: Layer
    qkv_proj
    qk_point_proj
    v_point_proj
    bias_proj
    out_proj
    head_weight
    num_heads::Int
    head_dim::Int
    num_query_points::Int
    num_point_values::Int
end

function InvariantPointAttention(;
    hidden_size::Int, num_heads::Int,
    head_dim = hidden_size ÷ num_heads,
    num_query_points::Int, num_point_values::Int,
    pair_size::Int,
)
    return InvariantPointAttention(
        LinearNoBias(hidden_size => head_dim * num_heads * 3),
        LinearNoBias(hidden_size => 3 * num_heads * num_query_points * 2),
        LinearNoBias(hidden_size => 3 * num_heads * num_point_values),
        LinearNoBias(pair_size => num_heads),
        Linear(num_heads * (head_dim + pair_size + num_point_values * (3 + 1)) => hidden_size),
        log.(exp.(ones(Float32, num_heads)) .- 1), # 1 after softplus
        num_heads, head_dim, num_query_points, num_point_values
    )
end

function (layer::InvariantPointAttention)(s, z, (R, t))
    q, k, v = split_axis(layer.qkv_proj(s), 3; dims=1)
    q, k, v = rearrange.((q, k, v), einops"(c h) l ... -> c l h ..."; h=layer.num_heads)
    qᵖ, kᵖ = split_axis(layer.qk_point_proj(s), 2; dims=1)
    qᵖ, kᵖ = rearrange.((qᵖ, kᵖ), einops"(d h p) l ... -> d p l ... h"; h=layer.num_heads, p=layer.num_query_points) .|> as_vectors
    vᵖ = rearrange(layer.v_point_proj(s), einops"(d h p) l ... -> d p l h ..."; h=layer.num_heads, p=layer.num_point_values) |> as_vectors
    b = rearrange(layer.bias_proj(z), einops"h k q ... -> k q h ...")
    wC, wL = √(2 / (9 * layer.num_query_points)), 1 / √3
    wC, wL, c = map(x -> ofeltype(x, q), (wC, wL, layer.head_dim))
    qᵖᵢ, (Rq, tq) = rearrange(qᵖ, einops"p i ... -> p 1 i ..."), rearrange.((R, t), einops"i ... -> 1 1 i ...")
    kᵖⱼ, (Rk, tk) = rearrange(kᵖ, einops"p j ... -> p j 1 ..."), rearrange.((R, t), einops"j ... -> 1 j 1 ...")
    γ = rearrange(softplus.(layer.head_weight), einops"h -> 1 1 h")
    Σₚ′ = sum(sum.(abs2, (Rq .* qᵖᵢ .+ tq) .- (Rk .* kᵖⱼ .+ tk)), dims=1)
    Σₚ = rearrange(Σₚ′, einops"1 k q ... h -> k q h ...")
    a = softmax(wL .* ((k)ᵀ ⊗ q ./ √c .+ b .- (γ .* (wC / 2) .* Σₚ)), dims=1)
    õ = einsum(a, z, einops"j i h ..., c j i ... -> (c h) i ...")
    o = einsum(a, v, einops"j i h ..., c j h ... -> (c h) i ...")
    Rv, tv = rearrange.((R, t), einops"j ... -> 1 j 1 ...")
    Tvᵖ = as_array(Rv .* vᵖ .+ tv)
    ō = einsum(a, Tvᵖ, einops"j i ..., d p j ... -> d p i ...", d=3)
    ō = as_array(adjoint.(Rv) .* (as_vectors(ō) .- tv))
    ō_flat = rearrange(ō, einops"d p i h ... -> (d h p) i ...")
    ō_norm = reduce(_norm, ō_flat, einops"(d hp) ... -> hp ...", d=3)
    s̃ = vcat(õ, o, ō_flat, ō_norm)
    return layer.out_proj(s̃)
end
