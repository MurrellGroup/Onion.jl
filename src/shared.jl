"""
    glut(t::AbstractArray, d::Int, pos::Int)
    glut(t::Real, d::Int, pos::Int) = t

`glut` adds dimensions to the middle. The resulting array will have `d` dimensions. `pos` is where to add the dimensions.
`pos=0` adds dims to the start, `pos=1` after the first element, etc.
If `t` is scalar, it is returned unmodified (because scalars don't need to match dims to broadcast).

Typically when broadcasting `x .* t`, you would call something like `glut(t, ndims(x), 1)`.
"""
glut(t::Real, d::Int, pos::Int) = t

function glut(t::AbstractArray, d::Int, pos::Int)
    ndt = ndims(t)
    d - ndt < 0 && error("Cannot expand array of size $(size(t)) to $d dimensions.")
    reshape(t, size(t)[1:pos]..., ntuple(Returns(1), (d - ndt))..., size(t)[(pos+1):end]...)
end

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

const as_dense_on_device = like


ofeltype(v::Number, x::AbstractArray{T}) where T = convert(T, v)


function watmul(W::AbstractArray{T,3}, x::AbstractArray{T}; r=1) where T
    x′ = rearrange(x,  einops"(s1 k r) ... -> s1 k r ..."; k=size(W, 3), r)
    y′ = einsum(W, x′, einops"s2 s1 k, s1 k r ... -> s2 k r ...")::typeof(x′)
    y  = rearrange(y′, einops"s2 k r ... -> (s2 k r) ...")
    return y
end

const ⨝ = watmul
