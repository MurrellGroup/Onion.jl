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
    x1 = l.fc1(x)
    return l.fc3(swish.(x1) .* l.fc2(x))
end
