struct ESMFoldEmbedConfig
    c_s::Int
    c_z::Int
    use_esm_attn_map::Bool
end

@concrete struct LayerNormMLP <: Onion.Layer
    norm
    fc1
    fc2
end

@layer LayerNormMLP

function LayerNormMLP(in_dim::Int, out_dim::Int)
    norm = Onion.LayerNorm(in_dim; eps=1f-5)
    fc1 = Onion.Linear(in_dim => out_dim)
    fc2 = Onion.Linear(out_dim => out_dim)
    return LayerNormMLP(norm, fc1, fc2)
end

function (mlp::LayerNormMLP)(x)
    x = mlp.norm(x)
    x = mlp.fc1(x)
    x = max.(x, 0f0)
    x = mlp.fc2(x)
    return x
end
