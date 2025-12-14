"""
    Linear(
        d1 => d2, k, σ=identity;
        bias::Bool=true,
        init=Flux.glorot_uniform
    )

See also [`BlockLinear`](@ref).
"""
@concrete struct Linear <: Layer
    weight; bias; σ
end

function Linear(
    (d1, d2)::Pair{Int,Int}, σ=identity;
    bias::Bool=true, init=Flux.glorot_uniform
)
    W = init(d2, d1)
    b = bias ? zeros_like(W, d2) : false
    return Linear(W, b, σ)
end

# σ.(W * x .+ b)
function ((; weight, bias, σ)::Linear)(x)
    x′ = reshape(x, size(x, 1), :)# Keep(), :)
    y′ = weight * x′
    y = reshape(y′, size(y′, 1), size(x)[2:end]...)#Keep(), Split(1, size(x)[2:end]))
    NNlib.bias_act!(σ, y, bias)
    return y
end

function Base.show(io::IO, (; weight, bias, σ)::Linear)
    print(io, "Linear($(size(weight, 2)) => $(size(weight, 1))")
    σ == identity || print(io, ", $(σ)")
    bias isa Union{Nothing,Bool} && print(io, ", bias=false")
    print(io, ")")
end
