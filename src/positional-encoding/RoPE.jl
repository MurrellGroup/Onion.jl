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
h = t(h; rope=rope[1:seqlen]) #Note the subsetting to match seqlen
```
"""
struct RoPE{A<:AbstractArray}
    cos::A
    sin::A
end

@layer RoPE trainable=()

Base.getindex(rope::RoPE, i) = RoPE(selectdim(rope.cos, 2, i), selectdim(rope.sin, 2, i))

function apply_scaling!(freqs::AbstractVector; scale_factor=8)
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    for (i, freq) in enumerate(freqs)
        wavelen = 2π / freq
        if wavelen > low_freq_wavelen
            freqs[i] = freq / scale_factor
        elseif wavelen > high_freq_wavelen
            @assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / 
                    (high_freq_factor - low_freq_factor)
            freqs[i] = (1 - smooth) * freq / scale_factor + smooth * freq
        end
    end
    return freqs
end

function RoPE(
    dim::Int, end_pos::Int; 
    T=Float32,
    theta=10000f0, use_scaled=false, scale_factor=8, start_pos=0
)
    theta = T(theta)
    freqs = 1f0 ./ (theta .^ (T.(0:2:dim-1)[1:dim÷2] ./ dim))
    use_scaled && apply_scaling!(freqs; scale_factor)
    freqs_complex = transpose(cis.(T.(start_pos:end_pos-1) * freqs'))
    return RoPE(real(freqs_complex), imag(freqs_complex))
end

# Note about Huggingface weights and rotary embeddings:
# https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509
# Use this one if you're using the Hugging Face weights.
function (rope::RoPE)(x)
    head_dim = size(x, 1)
    x1 = selectdim(x, 1, 1:head_dim÷2)
    x2 = selectdim(x, 1, head_dim÷2+1:size(x, 1))
    return vcat(
        x1 .* rope.cos .- x2 .* rope.sin,
        x2 .* rope.cos .+ x1 .* rope.sin
    )
end
