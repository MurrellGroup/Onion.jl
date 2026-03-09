using Flux
using NNlib

# BoltzGen triangle projections use the shared combine_projections_forward dispatch hook
# (defined in protein/triangular.jl) which routes to GPU kernels when available.

@concrete struct MiniTriangularUpdate <: Layer
    norm_in
    p_in
    g_in
    norm_out
    p_out
    g_out
end

@layer MiniTriangularUpdate

function MiniTriangularUpdate(dim::Int=128)
    norm_in = BGLayerNorm(dim; eps=1f-5)
    p_in = Dense(dim => dim, bias=false)
    g_in = Dense(dim => dim, bias=false)

    norm_out = BGLayerNorm(dim ÷ 2; eps=1f-5)
    p_out = Dense(dim ÷ 2 => dim, bias=false)
    g_out = Dense(dim ÷ 2 => dim, bias=false)

    bias_init_one!(norm_in.w)
    bias_init_zero!(norm_in.b)
    lecun_normal_init!(p_in.weight)
    gating_init!(g_in.weight)

    bias_init_one!(norm_out.w)
    bias_init_zero!(norm_out.b)
    final_init!(p_out.weight)
    gating_init!(g_out.weight)

    return MiniTriangularUpdate(norm_in, p_in, g_in, norm_out, p_out, g_out)
end

function (layer::MiniTriangularUpdate)(x; mask, use_kernels::Bool=false)
    d = size(x, 1)
    @assert eltype(x) === eltype(layer.p_in.weight) "MiniTriangularUpdate input eltype $(eltype(x)) must match weight eltype $(eltype(layer.p_in.weight))"

    x = layer.norm_in(x)
    x = layer.p_in(x) .* NNlib.sigmoid.(layer.g_in(x))

    mask_b = reshape(mask, 1, size(mask, 1), size(mask, 2), size(mask, 3))
    x = x .* mask_b

    @assert d % 4 == 0
    chunk = d ÷ 4

    a1 = view(x, 1:chunk, :, :, :)
    b1 = view(x, chunk+1:2*chunk, :, :, :)
    a2 = view(x, 2*chunk+1:3*chunk, :, :, :)
    b2 = view(x, 3*chunk+1:4*chunk, :, :, :)

    x1 = combine_projections_forward(a1, b1, true)
    x2 = combine_projections_forward(a2, b2, false)

    x = cat(x1, x2; dims=1)

    x = layer.norm_out(x)
    x = layer.p_out(x) .* NNlib.sigmoid.(layer.g_out(x))
    return x
end

@concrete struct BGTriangleMultiplicationOutgoing <: Layer
    norm_in
    p_in
    g_in
    norm_out
    p_out
    g_out
end

@layer BGTriangleMultiplicationOutgoing

function BGTriangleMultiplicationOutgoing(dim::Int=128)
    norm_in = BGLayerNorm(dim; eps=1f-5)
    p_in = Dense(dim => 2*dim, bias=false)
    g_in = Dense(dim => 2*dim, bias=false)

    norm_out = BGLayerNorm(dim; eps=1f-5)
    p_out = Dense(dim => dim, bias=false)
    g_out = Dense(dim => dim, bias=false)

    bias_init_one!(norm_in.w)
    bias_init_zero!(norm_in.b)
    lecun_normal_init!(p_in.weight)
    gating_init!(g_in.weight)

    bias_init_one!(norm_out.w)
    bias_init_zero!(norm_out.b)
    final_init!(p_out.weight)
    gating_init!(g_out.weight)

    return BGTriangleMultiplicationOutgoing(norm_in, p_in, g_in, norm_out, p_out, g_out)
end

function (layer::BGTriangleMultiplicationOutgoing)(x; mask, use_kernels::Bool=false)
    @assert eltype(x) === eltype(layer.p_in.weight) "BGTriangleMultiplicationOutgoing input eltype $(eltype(x)) must match weight eltype $(eltype(layer.p_in.weight))"
    x = layer.norm_in(x)
    x_in = x
    x = layer.p_in(x) .* NNlib.sigmoid.(layer.g_in(x))

    mask_b = reshape(mask, 1, size(mask, 1), size(mask, 2), size(mask, 3))
    x = x .* mask_b

    d = size(x_in, 1)
    a = view(x, 1:d, :, :, :)
    b = view(x, d+1:2*d, :, :, :)

    x = combine_projections_forward(a, b, true)

    x = layer.p_out(layer.norm_out(x)) .* NNlib.sigmoid.(layer.g_out(x_in))
    return x
end

@concrete struct BGTriangleMultiplicationIncoming <: Layer
    norm_in
    p_in
    g_in
    norm_out
    p_out
    g_out
end

@layer BGTriangleMultiplicationIncoming

function BGTriangleMultiplicationIncoming(dim::Int=128)
    norm_in = BGLayerNorm(dim; eps=1f-5)
    p_in = Dense(dim => 2*dim, bias=false)
    g_in = Dense(dim => 2*dim, bias=false)

    norm_out = BGLayerNorm(dim; eps=1f-5)
    p_out = Dense(dim => dim, bias=false)
    g_out = Dense(dim => dim, bias=false)

    bias_init_one!(norm_in.w)
    bias_init_zero!(norm_in.b)
    lecun_normal_init!(p_in.weight)
    gating_init!(g_in.weight)

    bias_init_one!(norm_out.w)
    bias_init_zero!(norm_out.b)
    final_init!(p_out.weight)
    gating_init!(g_out.weight)

    return BGTriangleMultiplicationIncoming(norm_in, p_in, g_in, norm_out, p_out, g_out)
end

function (layer::BGTriangleMultiplicationIncoming)(x; mask, use_kernels::Bool=false)
    @assert eltype(x) === eltype(layer.p_in.weight) "BGTriangleMultiplicationIncoming input eltype $(eltype(x)) must match weight eltype $(eltype(layer.p_in.weight))"
    x = layer.norm_in(x)
    x_in = x
    x = layer.p_in(x) .* NNlib.sigmoid.(layer.g_in(x))

    mask_b = reshape(mask, 1, size(mask, 1), size(mask, 2), size(mask, 3))
    x = x .* mask_b

    d = size(x_in, 1)
    a = view(x, 1:d, :, :, :)
    b = view(x, d+1:2*d, :, :, :)

    x = combine_projections_forward(a, b, false)

    x = layer.p_out(layer.norm_out(x)) .* NNlib.sigmoid.(layer.g_out(x_in))
    return x
end
