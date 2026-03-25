using NNlib: swish

"""
    Transition(dim, hidden=4*dim; out_dim=dim)

BoltzGen-style SwiGLU feed-forward: `fc3(swish(fc1(norm(x))) .* fc2(norm(x)))`.
"""
@concrete struct Transition <: Layer
    norm; fc1; fc2; fc3
end

function Transition(dim::Int, hidden::Int=4*dim; out_dim::Int=dim)
    norm = LayerNorm(dim)
    fc1  = Linear(dim => hidden, bias=false)
    fc2  = Linear(dim => hidden, bias=false)
    fc3  = Linear(hidden => out_dim, bias=false)
    return Transition(norm, fc1, fc2, fc3)
end

function (l::Transition)(x)
    x = l.norm(x)
    x′ = reshape(x, Keep(), :)
    y′ = glu_ffn(x′, l.fc1.weight, l.fc2.weight, l.fc3.weight, swish)
    return reshape(y′, Keep(), Split(.., Base.tail(size(x))))
end
