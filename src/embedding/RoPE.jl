"""
    RoPE(dim::Int, max_length; theta::T=10000f0)

Rotary Position Embeddings (as in Llama3).
    
```julia
dim = 64
n_heads = 8
n_kv_heads = 4
seqlen = 10

t = TransformerBlock(dim, n_heads, n_kv_heads)
h = randn(Float32, dim, seqlen, 1)

rope = RoPE(dim ÷ n_heads, 1000)
h = t(h, 1, rope[1:seqlen]) #Note the subsetting to match seqlen
```
"""
struct RoPE{A<:AbstractArray}
    cos::A
    sin::A
end

Flux.@layer RoPE trainable=()

Base.getindex(rope::RoPE, i) = @views RoPE(rope.cos[:,i,:,:], rope.sin[:,i,:,:])

function RoPE(dim::Int, end_pos::Int; theta::T=10000f0, start_pos=0) where T
    freqs = 1f0 ./ (theta .^ (T.(0:2:dim-1)[1:dim÷2] ./ dim))
    freqs_complex = cis.(T.(start_pos:end_pos-1) * freqs')
    cos = permutedims(real(freqs_complex), (2, 1))  # (head_dim/2, seq_len)
    sin = permutedims(imag(freqs_complex), (2, 1))
    cos = reshape(cos, (dim÷2, end_pos - start_pos, 1, 1))
    sin = reshape(sin, (dim÷2, end_pos - start_pos, 1, 1))
    return RoPE(cos, sin)
end

# Note about Huggingface weights and rotary embeddings:
# https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509
# Use this one if you're using the Hugging Face weights.
function (rope::RoPE)(x)
    head_dim = size(x, 1)
    x1 = x[1:head_dim÷2, :, :, :]
    x2 = x[head_dim÷2+1:end, :, :, :]
    return vcat(  
        x1 .* rope.cos .- x2 .* rope.sin,
        x2 .* rope.cos .+ x1 .* rope.sin
    )
end
