"""
    Linear(
        d1 => d2, k, σ=identity;
        bias::Bool=true,
        init=Flux.glorot_uniform
    )

See also [`BlockLinear`](@ref).
"""
@concrete struct Linear
    weight <: AbstractArray
    bias <: Maybe{AbstractArray}
    σ
end

@layer Linear

function Linear(
    (d1, d2)::Pair{Int,Int}, σ=identity;
    bias::Bool=true, init=Flux.glorot_uniform
)
    W = init(d2, d1)
    b = bias ? zeros_like(W, d2) : nothing
    return Linear(W, b, σ)
end

# σ.(W * x .+ b)
function ((; weight, bias, σ)::Linear)(x)
    x′ = hardreshape(x, size(x, 1), :)
    y′ = weight * x′
    NNlib.bias_act!(σ, y′, @something bias false)
    y = reshape(y′, size(y′, 1), size(x)[2:end]...)
    return y
end
