struct KVCache{T,A<:AbstractArray{T}}
    k::A
    v::A
    pos::Ref{Int}
end

function KVCache(k::A, v::A, pos::Int=0) where {T,A<:AbstractArray{T,4}}
    size(k) == size(v) || throw(DimensionMismatch("k and v must have the same size"))
    return KVCache(k, v, Ref(pos))
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
    k = similar(layer.wq.weight, layer.head_dim, len, layer.n_kv_heads, batch)
    v = similar(layer.wv.weight, layer.head_dim, len, layer.n_kv_heads, batch)
    return KVCache(k, v)
end

function kv_cache(layer::TransformerBlock, len::Int, batch::Int=1)
    return kv_cache(layer.attention, len, batch)
end

function extend(cache::KVCache, new_len::Int)
    head_dim, len, kv_heads, batch = size(cache.k)
    @assert new_len > len
    k = similar(cache.k, head_dim, new_len, kv_heads, batch)
    v = similar(cache.v, head_dim, new_len, kv_heads, batch)
    k[:, 1:len, :, :] .= cache.k
    v[:, 1:len, :, :] .= cache.v
    return KVCache(k, v, cache.pos)
end

function (cache::KVCache)(k::AbstractArray, v::AbstractArray)
    cache.k[:, pos(cache) .+ axes(k, 2), :, :] .= k
    cache.v[:, pos(cache) .+ axes(v, 2), :, :] .= v
    pos!(cache, pos(cache) + size(k, 2))
    b = ndims(k) == ndims(v) == 3 ? 1 : (:)
    return @views cache.k[:, 1:pos(cache), :, b], cache.v[:, 1:pos(cache), :, b]
end
