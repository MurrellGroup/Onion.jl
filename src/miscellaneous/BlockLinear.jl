"""
    BlockLinear(
        d1 => d2, k, σ=identity;
        bias::Bool=true, init=Flux.glorot_uniform)

A block-diagonal version of a linear layer, comprising `k` blocks,
where the blocks are of size `(d2 ÷ k, d1 ÷ k)`.

Equivalent to [`Linear`](@ref) when `k=1`.
"""
@concrete struct BlockLinear <: Layer
    weight <: AbstractArray
    bias <: Maybe{AbstractArray}
    σ
end

input_size(bl::BlockLinear) = size(bl.weight, 2) * size(bl.weight, 3)
output_size(bl::BlockLinear) = size(bl.weight, 1) * size(bl.weight, 3)

function BlockLinear(
    (d1, d2)::Pair{Int,Int}, k::Int, σ=identity;
    bias::Bool=true, init=Flux.glorot_uniform
)
    d1 % k == 0 || throw(ArgumentError("d1 must be divisible by k"))
    d2 % k == 0 || throw(ArgumentError("d2 must be divisible by k"))
    s1, s2 = d1 ÷ k, d2 ÷ k
    W = init(s2, s1, k)
    b = bias ? zeros_like(W, d2) : false
    return BlockLinear(W, b, σ)
end

# σ.(W ⨝ x .⊞ b)
function ((; weight, bias, σ)::BlockLinear)(x)
    y = weight ⨝ x
    NNlib.bias_act!(σ, y, @something bias false)
    return y
end

function Base.show(io::IO, bl::BlockLinear)
    (; weight, bias, σ) = bl
    k = size(weight, 3)
    print(io, "BlockLinear($(input_size(bl)) => $(output_size(bl)), $k")
    σ == identity || print(io, ", $(σ)")
    bias isa Union{Nothing,Bool} && print(io, ", bias=false")
    print(io, ")")
end
