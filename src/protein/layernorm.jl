using Statistics

# Dispatch function for OnionTile to override with fused GPU kernel
layernorm_first_forward(x, w, b; eps) = _cpu_layernorm_first(x, w, b; eps)

function _cpu_layernorm_first(x::AbstractArray, w, b; eps)
    shape = ntuple(_ -> 1, ndims(x) - 1)
    w_r = reshape(w, length(w), shape...)
    b_r = reshape(b, length(b), shape...)
    μ = Statistics.mean(x; dims=1)
    diff = x .- μ
    σ2 = Statistics.mean(diff .* diff; dims=1)
    inv_std = 1f0 ./ sqrt.(σ2 .+ eps)
    return @. diff * inv_std * w_r + b_r
end

@concrete struct LayerNormFirst <: Onion.Layer
    w
    b
    eps::Float32
end

@layer LayerNormFirst

function LayerNormFirst(dim::Int; eps=1f-5)
    w = ones(Float32, dim)
    b = zeros(Float32, dim)
    return LayerNormFirst(w, b, Float32(eps))
end

function (ln::LayerNormFirst)(x::AbstractArray)
    return layernorm_first_forward(x, ln.w, ln.b; eps=ln.eps)
end

# In-place LayerNorm: default copies result back. OnionTile overrides with fused kernel.
function layernorm_inplace!(ln::LayerNormFirst, x::AbstractArray)
    result = ln(x)
    copyto!(x, result)
    return x
end

@concrete struct LinearFirst <: Onion.Layer
    weight
    bias
    use_bias::Bool
end

@layer LinearFirst

function LinearFirst(in_dim::Int, out_dim::Int; bias::Bool=true)
    scale = Float32(1 / sqrt(Float32(in_dim)))
    weight = randn(Float32, out_dim, in_dim) .* scale
    b = bias ? zeros(Float32, out_dim) : zeros(Float32, 0)
    return LinearFirst(weight, b, bias)
end

function (m::LinearFirst)(x::AbstractArray)
    in_dim = size(m.weight, 2)
    out_dim = size(m.weight, 1)
    x2 = reshape(x, in_dim, :)
    y2 = m.weight * x2
    if m.use_bias
        y2 .+= m.bias
    end
    out_shape = (out_dim, size(x)[2:end]...)
    return reshape(y2, out_shape)
end
