using NNlib

# BGLayerNorm is mathematically identical to LayerNormFirst (verified: max diff 4.77e-7).
# Using LayerNormFirst gives access to the layernorm_first_forward GPU dispatch hook.
# Checkpoint loading handles field name mapping (weight→w, bias→b) automatically.
const BGLayerNorm = LayerNormFirst

@concrete struct BGLinear <: Layer
    weight
    bias
end

@layer BGLinear

function BGLinear(
    in_dim::Int,
    out_dim::Int;
    bias::Bool=true,
    init::Symbol=:default,
)
    weight = zeros(Float32, out_dim, in_dim)
    if init === :default
        lecun_normal_init!(weight)
    elseif init === :relu
        he_normal_init!(weight)
    elseif init === :glorot
        glorot_uniform_init!(weight)
    elseif init === :gating
        gating_init!(weight)
    elseif init === :normal
        normal_init!(weight)
    elseif init === :final
        final_init!(weight)
    else
        error("Invalid init string: $init")
    end

    bias_arr = bias ? zeros(Float32, out_dim) : nothing
    if bias && init === :gating
        bias_init_one!(bias_arr)
    end
    return BGLinear(weight, bias_arr)
end

function (l::BGLinear)(x)
    in_dim = size(l.weight, 2)
    x2 = reshape(x, in_dim, :)
    y = l.weight * x2
    if l.bias !== nothing
        y .+= reshape(eltype(x).(l.bias), :, 1)
    end
    return reshape(y, size(l.weight, 1), size(x)[2:end]...)
end

