const SiLU = NNlib.swish

@concrete struct MultiHeadFFN <: Layer
    in_proj   # W_in
    up_proj   # U
    gate_proj # K
    down_proj # V
    out_proj  # W_out
    num_heads::Int
end

function MultiHeadFFN(
    hidden_size::Integer,
    intermediate_size::Integer,
    num_heads::Integer,
    head_dim::Integer = hidden_size ÷ num_heads;
    init = Flux.glorot_uniform,
)
    MultiHeadFFN(
        Dense(hidden_size => head_dim * num_heads, bias=false),
        init(intermediate_size, head_dim, num_heads),
        init(intermediate_size, head_dim, num_heads),
        init(head_dim, intermediate_size, num_heads),
        Dense(head_dim * num_heads => hidden_size, bias=false),
        num_heads
    )
end

function (layer::MultiHeadFFN)(x::AbstractArray)
    q = layer.in_proj(x)
    q = rearrange(q, einops"(d h) ... -> d h ..."; h=layer.num_heads)
    s = naive_mhf(q, layer.gate_proj, layer.up_proj, layer.down_proj)
    s = rearrange(s, einops"d h ... -> (d h) ...")
    o = layer.out_proj(s)
    return o
end

function naive_mhf(q, k, u, v)
    ϕ = SiLU.(einsum(k, q, einops"i dₕ h, dₕ h ... -> i h ...")) .* einsum(u, q, einops"i dₕ h, dₕ h ... -> i h ...")
    s = einsum(v, ϕ, einops"dₕ i h, i h ... -> dₕ h ...")
    return s
end
