"""
    Linear(
        d1 => d2;
        bias::Bool=true,
        init=Flux.glorot_uniform
    )

See also [`BlockLinear`](@ref).
"""
@concrete struct Linear <: Layer
    weight; bias
end

function Linear(;
    in_size::Int, out_size::Int,
    bias::Bool=true, init=Flux.glorot_uniform
)
    W = init(d2, d1)
    b = bias ? zeros_like(W, d2) : false
    return Linear(W, b)
end

Linear((d1, d2)::Pair{Int,Int}; kws...) = Linear(d1, d2; kws...)

# σ.(W * x .+ b)
function ((; weight, bias)::Linear)(x)
    x′ = reshape(x, Keep(), :)
    y′ = weight * x′
    y = reshape(y′, Keep(), Split(.., size(x)[2:end]))
    NNlib.bias_act!(identity, y, @something bias false)
    return y
end

function Base.show(io::IO, (; weight, bias)::Linear)
    print(io, "Linear($(size(weight, 2)) => $(size(weight, 1))")
    bias isa Union{Nothing,Bool} && print(io, ", bias=false")
    print(io, ")")
end
