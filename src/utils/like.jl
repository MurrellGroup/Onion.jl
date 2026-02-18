using Random

"""
    like(v, x::AbstractArray, [T=eltype(x)], [dims=size(x)])

Returns an array of `v` (converted to type `T`) with an array type similar to `x`.
The element type and dimensions default to `eltype(x)` and `size(x)`.

`like(v, x::AbstractArray, args...)` is equivalent to `fill!(similar(x, args...), v)`,
but the function is marked as non-differentiable using `ChainRulesCore`.
"""
like(v, x::AbstractArray, args...) = @ignore_derivatives fill!(similar(x, args...), v)

"""
    zeros_like(x::AbstractArray, [T=eltype(x)], [dims=size(x)])

Returns an array of zeros with an array type similar to `x`.
The element type and dimensions default to `eltype(x)` and `size(x)`.

`zeros_like(args...)` is equivalent to `like(false, args...)`
"""
zeros_like(x::AbstractArray, args...) = like(false, x, args...)

"""
    ones_like(x::AbstractArray, [T=eltype(x)], [dims=size(x)])

Returns an array of ones with an array type similar to `x`.
The element type and dimensions default to `eltype(x)` and `size(x)`.

`ones_like(args...)` is equivalent to `like(true, args...)`
"""
ones_like(x::AbstractArray, args...) = like(true, x, args...)

"""
    falses_like(x::AbstractArray, [T=eltype(x)], [dims=size(x)])

Returns an array of falses of type `Bool` with an array type similar to `x`.
The dimensions default to `size(x)`.

`falses_like(args...)` is equivalent to `like(false, Bool, args...)`
"""
falses_like(x::AbstractArray, args...) = zeros_like(x, Bool, args...)

"""
    trues_like(x::AbstractArray, [T=eltype(x)], [dims=size(x)])

Returns an array of trues of type `Bool` with an array type similar to `x`.
The dimensions default to `size(x)`.

`trues_like(args...)` is equivalent to `like(true, Bool, args...)`
"""
trues_like(x::AbstractArray, args...) = ones_like(x, Bool, args...)

_random_like(f!, x, rng::AbstractRNG, args...) = @ignore_derivatives f!(rng, similar(x, args...))
_random_like(f!, x, args...) = _random_like(f!, x, Random.default_rng(), args...)

rand_like(x::AbstractArray, args...) = _random_like(rand!, x, args...)
randn_like(x::AbstractArray, args...) = _random_like(randn!, x, args...)

"""
    like(x::AbstractArray, array::DenseArray, T=eltype(x))

Like `like(v, x::AbstractArray, args...)`, but an arbitrary `AbstractArray`, such as an `AbstractRange`,
can be instantiated on device.

# Examples

```jldoctest
julia> like(1:5, rand(1))
5-element Vector{Int64}:
 1
 2
 3
 4
 5

julia> like((1:5)', rand(1), Float32)
1×5 Matrix{Float32}:
 1.0  2.0  3.0  4.0  5.0
```
"""
like(x::AbstractArray, array::DenseArray, T=eltype(x)) = @ignore_derivatives similar(array, T, size(x)) .= x
