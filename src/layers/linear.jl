"""
    Linear(
        d1 => d2;
        bias::Bool=true,
        init=WeightInitializers.glorot_uniform
    )

See also [`BlockLinear`](@ref).
"""
@concrete struct Linear <: Layer
    weight; bias
end

function Linear(;
    in_size::Int, out_size::Int,
    bias::Bool=true, init=with_default_rng(WI.glorot_uniform)
)
    W = init(out_size, in_size)
    b = bias ? zeros_like(W, out_size) : false
    return Linear(W, b)
end

Linear((d1, d2)::Pair{Int,Int}; kws...) = Linear(; in_size=d1, out_size=d2, kws...)

function forward(l::Linear, r::Rules, x)
    x′ = reshape(x, Keep(), :)
    y′ = linear(r, x′, l.weight, l.bias)
    return reshape(y′, Keep(), Split(.., Base.tail(size(x))))
end

function Base.show(io::IO, (; weight, bias)::Linear)
    print(io, "Linear($(size(weight, 2)) => $(size(weight, 1))")
    bias isa Union{Nothing,Bool} && print(io, ", bias=false")
    print(io, ")")
end


"""
    BlockLinear(
        d1 => d2, k;
        bias::Bool=true,
        init=WeightInitializers.glorot_uniform)

A block-diagonal version of a linear layer, comprising `k` blocks,
where the blocks are of size `(d2 ÷ k, d1 ÷ k)`.

Equivalent to [`Linear`](@ref) when `k=1`.
"""
@concrete struct BlockLinear <: Layer
    weight <: AbstractArray
    bias <: Optional{AbstractArray}
end

input_size(bl::BlockLinear) = size(bl.weight, 2) * size(bl.weight, 3)
output_size(bl::BlockLinear) = size(bl.weight, 1) * size(bl.weight, 3)

function BlockLinear(
    (d1, d2)::Pair{Int,Int}, k::Int;
    bias::Bool=true, init=with_default_rng(WI.glorot_uniform)
)
    d1 % k == 0 || throw(ArgumentError("d1 must be divisible by k"))
    d2 % k == 0 || throw(ArgumentError("d2 must be divisible by k"))
    s1, s2 = d1 ÷ k, d2 ÷ k
    W = init(s2, s1, k)
    b = bias ? zeros_like(W, d2) : false
    return BlockLinear(W, b)
end

# σ.(W ⨝ x .⊞ b)
function ((; weight, bias)::BlockLinear)(x)
    y = weight ⨝ x
    NNlib.bias_act!(identity, y, @something bias false)
    return y
end

function Base.show(io::IO, bl::BlockLinear)
    (; weight, bias) = bl
    k = size(weight, 3)
    print(io, "BlockLinear($(input_size(bl)) => $(output_size(bl)), $k")
    bias isa Union{Nothing,Bool} && print(io, ", bias=false")
    print(io, ")")
end
