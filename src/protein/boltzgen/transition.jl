using Flux
using NNlib

@concrete struct Transition <: Layer
    norm
    fc1
    fc2
    fc3
    hidden::Int
end

@layer Transition

function Transition(dim::Int=128, hidden::Int=512; out_dim::Union{Nothing,Int}=nothing)
    out_dim = isnothing(out_dim) ? dim : out_dim

    norm = BGLayerNorm(dim; eps=1f-5)
    fc1 = Dense(dim => hidden, bias=false)
    fc2 = Dense(dim => hidden, bias=false)
    fc3 = Dense(hidden => out_dim, bias=false)

    bias_init_one!(norm.w)
    bias_init_zero!(norm.b)

    lecun_normal_init!(fc1.weight)
    lecun_normal_init!(fc2.weight)
    final_init!(fc3.weight)

    return Transition(norm, fc1, fc2, fc3, hidden)
end

function (layer::Transition)(x; chunk_size::Union{Nothing,Int}=nothing)
    x = layer.norm(x)

    if isnothing(chunk_size) || bg_istraining()
        x1 = layer.fc1(x)
        x = (x1 .* NNlib.sigmoid.(x1)) .* layer.fc2(x)
        return layer.fc3(x)
    end

    hidden = layer.hidden
    out = nothing
    x_flat = reshape(x, size(x, 1), :)
    for i in 1:chunk_size:hidden
        hi = i:min(i + chunk_size - 1, hidden)
        w1 = layer.fc1.weight[hi, :]
        w2 = layer.fc2.weight[hi, :]
        w3 = layer.fc3.weight[:, hi]
        x_chunk = w1 * x_flat
        x_chunk = x_chunk .* NNlib.sigmoid.(x_chunk)
        x_chunk = x_chunk .* (w2 * x_flat)
        x_chunk = reshape(x_chunk, length(hi), size(x)[2:end]...)
        chunk_out = w3 * reshape(x_chunk, length(hi), :)
        chunk_out = reshape(chunk_out, size(w3, 1), size(x)[2:end]...)
        if out === nothing
            out = chunk_out
        else
            out = out .+ chunk_out
        end
    end
    return out
end
