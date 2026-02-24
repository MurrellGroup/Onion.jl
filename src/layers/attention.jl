# ──── Attention ────

"""
    Attention(
        in_dim::Int, n_heads::Int, n_kv_heads=n_heads;
        head_dim=in_dim÷n_heads, qkv_bias=false,
        q_norm=identity, k_norm=identity,
        out_init_scale=1,
    )

Attention layer that supports both self-attention and cross-attention (as in Llama3).

# Examples

## Self-attention

```julia
in_dim = 256
n_heads = 8
n_kv_heads = 4
head_dim = 64
attn = Attention(in_dim, n_heads, n_kv_heads; head_dim)

seq_len = 10
batch = 2
x = randn(in_dim, seq_len, batch)
output = attn(x)
```
"""
@concrete struct Attention <: Layer
    wq; wk; wv; wo
    q_norm; k_norm; g1_gate
    in_dim::Int
    head_dim::Int
    n_heads::Int
    n_kv_heads::Int
end

function Attention(
    in_dim::Int, n_heads::Int, n_kv_heads::Int=n_heads;
    head_dim = in_dim ÷ n_heads, qkv_bias=false,
    qk_norm=false,
    q_norm=qk_norm ? RMSNorm(head_dim) : identity,
    k_norm=qk_norm ? RMSNorm(head_dim) : identity,
    g1_gate=something,
    out_init_scale=1,
)
    @assert n_heads % n_kv_heads == 0 "n_heads must be divisible by n_kv_heads"
    wq = Linear(in_dim => n_heads * head_dim, bias=qkv_bias)
    wk = Linear(in_dim => n_kv_heads * head_dim, bias=qkv_bias)
    wv = Linear(in_dim => n_kv_heads * head_dim, bias=qkv_bias)
    wo = Linear(n_heads * head_dim => in_dim, bias=false)
    wo.weight .*= out_init_scale
    return Attention(wq, wk, wv, wo, q_norm, k_norm, g1_gate,
        in_dim, head_dim, n_heads, n_kv_heads)
end

function (layer::Attention)(
    xq, xk=xq, xv=xk;
    rope=identity, krope=rope, cache=tuple,
    kws...
)
    q, k, v = layer.wq(xq), layer.wk(xk), layer.wv(xv)
    q, k, v = rearrange.((q, k, v), einops"(d h) l ... -> d l h ..."; d=layer.head_dim)
    q, k = layer.q_norm(q), layer.k_norm(k)
    q, k = rope(q), krope(k)
    k, v = cache(k, v)
    k, v = repeat.((k, v), einops"d l h ... -> d l (r h) ..."; r=layer.n_heads÷layer.n_kv_heads)
    x = attention(q, k, v; kws...)
    x = rearrange(x, einops"d l h ... -> (d h) l ..."; h=layer.n_heads)
    x = layer.g1_gate(x, xq)
    return layer.wo(x)
end


# ──── KVCache ────

struct KVCache{T,A<:AbstractArray{T,4}}
    k::A
    v::A
    pos::Ref{Int}
end

function KVCache(k::A, v::A, pos::Int=0) where {T,A<:AbstractArray{T,4}}
    size(k) == size(v) || throw(DimensionMismatch("k and v must have the same size"))
    return KVCache{T,A}(k, v, Ref(pos))
end

Base.size(cache::KVCache, args...) = size(cache.k, args...)
Base.length(cache::KVCache) = size(cache, 2)
batch_size(cache::KVCache) = size(cache, 4)

pos(cache::KVCache) = cache.pos[]

function pos!(cache::KVCache, pos::Int)
    pos ≤ length(cache) || throw(BoundsError(cache, pos))
    cache.pos[] = pos
    return cache
end

function Base.show(io::IO, ::MIME"text/plain", cache::KVCache)
    println(io, typeof(cache), ':')
    println(io, "  size: $(size(cache.k))")
    print(io, "  position: $(pos(cache)) / $(length(cache))")
end

function kv_cache(layer::Attention, len::Int, batch::Int=1)
    k = zeros_like(layer.wq.weight, layer.head_dim, len, layer.n_kv_heads, batch)
    v = zeros_like(layer.wv.weight, layer.head_dim, len, layer.n_kv_heads, batch)
    return KVCache(k, v)
end

function extend(cache::KVCache, new_len::Int)
    head_dim, len, kv_heads, batch = size(cache.k)
    @assert new_len > len
    k = zeros_like(cache.k, head_dim, new_len, kv_heads, batch)
    v = zeros_like(cache.v, head_dim, new_len, kv_heads, batch)
    k[:, 1:len, :, :] .= cache.k
    v[:, 1:len, :, :] .= cache.v
    return KVCache(k, v, cache.pos)
end

function (cache::KVCache)(k::AbstractArray, v::AbstractArray)
    cache.k[:, pos(cache) .+ axes(k, 2), :, :] .= k
    cache.v[:, pos(cache) .+ axes(v, 2), :, :] .= v
    pos!(cache, pos(cache) + size(k, 2))
    b = ndims(k) == ndims(v) == 3 ? 1 : (:)
    return cache.k[:, 1:pos(cache), :, b], cache.v[:, 1:pos(cache), :, b]
end


# ──── DART ────

"""
    DART(transformer; mask=:causal)

"Doubly Auto-Regressive Transformer" (DART) is a convenience layer wrapping a
transformer block that can be used to model auto-regressive data represented
along two dimensions.

!!! note
    The mask acts on the flattened tokens sequence.

# Examples

```julia
julia> dart = DART(TransformerBlock(64, 8));

julia> x = randn(Float32, 64, 4, 20);

julia> dart(x) |> size
(64, 4, 20)
```
"""
@concrete struct DART <: Layer
    transformer
end

function (dart::DART)(x::AbstractArray; pair_feats=nothing, kws...)
    h = rearrange(x, (:d, :K, :L, ..) --> (:d, (:K, :L), ..))
    r = size(x, 2)
    isnothing(pair_feats) || (pair_feats = repeat(pair_feats, einops"h ql kl ... -> h (r1 ql) (r2 kl) ...", r1=r, r2=r))
    return reshape(dart.transformer(h; pair_feats, kws...), size(x))
end
