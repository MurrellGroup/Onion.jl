"""
    BlockDense(
        d1 => d2, k;
        σ=identity, bias=true, init=Flux.glorot_uniform)

A block-diagonal version of a dense layer. Equivalent to `Flux.Dense` when `k=1`.
"""
@concrete struct BlockDense
    weight; bias; σ
end

@layer BlockDense

function BlockDense(
    (d1, d2)::Pair{Int,Int}, k::Int, σ=identity;
    init=Flux.glorot_uniform, bias=true
)
    d1 % k == 0 || throw(ArgumentError("d1 must be divisible by k"))
    d2 % k == 0 || throw(ArgumentError("d2 must be divisible by k"))
    s1, s2 = d1 ÷ k, d2 ÷ k
    W = init(s2, s1, k)
    b = bias ? zeros_like(W, d2) : false
    return BlockDense(W, b, σ)
end

function (bd::BlockDense)(x)
    W, b, σ = bd.weight, bd.bias, bd.σ
    return σ.(W ⨝ x .+ b)
end
