function _split(x::AbstractArray{<:Any,N}, ::Val{sections}, ::Val{dims}) where {N,sections,dims}
    d = size(x, dims)
    s, r = divrem(d, sections)
    r == 0 || error()
    return ntuple(
        i -> view(x, ntuple(j -> j == dims ? (s*(i-1)+1:s*i) : (:), Val(N))...),
        Val(sections))
end

function _split(x, inds::NTuple{N,Int}, ::Val{dims}) where {N,dims}
    d = size(x, dims)
    inds′ = (0, inds..., d)
    any(>(d), inds′) && error()
    return ntuple(Val(N+1)) do k
        i, j = inds′[k], inds′[k+1]
        view(x, ntuple(d -> d == dims ? (i+1:j) : (:), Val(ndims(x)))...)
    end
end

Base.@constprop :aggressive function split(x, sections::Int; dims::Int)
    return _split(x, Val(sections), Val(dims))
end

Base.@constprop :aggressive function split(x, inds::NTuple{N,Int}; dims::Int) where N
    return _split(x, inds, Val(dims))
end

const split_axis = split
