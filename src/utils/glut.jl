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
