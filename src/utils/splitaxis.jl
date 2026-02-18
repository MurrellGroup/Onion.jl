function _split(x::AbstractArray{<:Any,N}, ::Val{sections}, ::Val{dims}) where {N,sections,dims}
    d = size(x, dims)
    s, r = divrem(d, sections)
    r == 0 || error()
    return ntuple(
        i -> view(x, ntuple(j -> j == dims ? (s*(i-1)+1:s*i) : (:), Val(N))...),
        Val(sections))
end

function _split(x, sizes::NTuple{N,Int}, ::Val{dims}) where {N,dims}
    d = size(x, dims)
    cs = cumsum(sizes)
    cs[end] == d || error()
    inds = (0, cs...)
    return ntuple(Val(N)) do k
        i, j = inds[k], inds[k+1]
        view(x, ntuple(d -> d == dims ? (i+1:j) : (:), Val(ndims(x)))...)
    end
end

Base.@constprop :aggressive function splitaxis(x, sections::Int; dims::Int=1)
    return _split(x, Val(sections), Val(dims))
end

Base.@constprop :aggressive function splitaxis(x, inds::NTuple{N,Int}; dims::Int=1) where N
    return _split(x, inds, Val(dims))
end
