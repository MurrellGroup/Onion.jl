function _split(x::AbstractArray{<:Any,N}, ::Val{sections}, ::Val{dims}) where {N,sections,dims}
    d = size(x, dims)
    s, r = divrem(d, sections)
    r == 0 || error()
    return ntuple(
        i -> view(x, ntuple(j -> j == dims ? (s*(i-1)+1:s*i) : (:), Val(N))...),
        Val(sections))
end

Base.@constprop :aggressive function split_axis(x, sections::Int; dims::Int)
    return _split(x, Val(sections), Val(dims))
end
