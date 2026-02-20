"""
    StarGLU(dim::Int, ff_hidden_dim::Int; act=Flux.swish)

Gated Linear Unit with flexible activation function (default: `swish`, making it a SwiGLU layer as used in Llama3).
"""
@concrete struct StarGLU{F} <: Layer
    act; up; gate; down
end

function StarGLU(;
    hidden_size::Int, intermediate_size::Int, hidden_size_out::Int=dim,
    act = Flux.swish,
)
    up = Linear(hidden_size => intermediate_size, bias=false)
    gate = Linear(hidden_size => intermediate_size, bias=false)
    down = Linear(intermediate_size => hidden_size_out, bias=false)
    return StarGLU(act, up, gate, down)
end

StarGLU((d1, d2)::Pair{Int,Int}; kws...) = StarGLU(d1, d2; kws...)
StarGLU((d1, (d2, d3))::Pair{Int,Pair{Int,Int}}; kws...) = StarGLU(d1, d2, d3; kws...)

function (l::StarGLU)(x::AbstractArray)
    x′ = reshape(x, Keep(), :)
    y′ = Primitives.glu_ffn(
        x′,
        l.up.weight, l.gate.weight, l.down.weight,
        l.act
    )
    y = reshape(y′, Keep(), Split(.., Base.tail(size(x))))
    return y
end
