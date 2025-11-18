"""
    BlockLinear(
        d1 => d2, k, σ=identity;
        bias::Bool=true, init=Flux.glorot_uniform)

A block-diagonal version of a linear layer, comprising `k` blocks,
where the blocks are of size `(d2 ÷ k, d1 ÷ k)`.

Equivalent to [`Linear`](@ref) when `k=1`.
"""
@concrete struct BlockLinear
    weight <: AbstractArray
    bias <: Maybe{AbstractArray}
    σ
end

@layer BlockLinear

function BlockLinear(
    (d1, d2)::Pair{Int,Int}, k::Int, σ=identity;
    bias::Bool=true, init=Flux.glorot_uniform
)
    d1 % k == 0 || throw(ArgumentError("d1 must be divisible by k"))
    d2 % k == 0 || throw(ArgumentError("d2 must be divisible by k"))
    s1, s2 = d1 ÷ k, d2 ÷ k
    W = init(s2, s1, k)
    b = bias ? zeros_like(W, d2) : nothing
    return BlockLinear(W, b, σ)
end

# σ.(W ⨝ x .⊞ b)
function ((; weight, bias, σ)::BlockLinear)(x)
    y = weight ⨝ x
    NNlib.bias_act!(σ, y, @something bias false)
    return y
end
