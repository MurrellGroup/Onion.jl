using NNlib

function permute_final_dims(x::AbstractArray, order::NTuple{N,Int}) where {N}
    n = ndims(x)
    k = length(order)
    prefix = collect(1:(n - k))
    # order is 0-based, match OpenFold's semantics
    tail = [n - k + o + 1 for o in order]
    return permutedims(x, vcat(prefix, tail))
end

function flatten_final_dims(x::AbstractArray, k::Int)
    n = ndims(x)
    @assert k <= n
    new_shape = (size(x)[1:(n - k)]..., :)
    return reshape(x, new_shape)
end

function dict_multimap(fn, dicts::AbstractVector{<:AbstractDict})
    first = dicts[1]
    out = Dict{Symbol,Any}()
    for (k, v) in first
        vals = [d[k] for d in dicts]
        if v isa AbstractDict
            out[k] = dict_multimap(fn, vals)
        else
            out[k] = fn(vals)
        end
    end
    return out
end

function stack_dicts(dicts::AbstractVector{<:AbstractDict})
    return dict_multimap(x -> stack(x; dims=1), dicts)
end

function one_hot_last(idx::AbstractArray, num_classes::Int)
    cls = reshape(0:(num_classes - 1), ntuple(_ -> 1, ndims(idx))..., num_classes)
    cls = to_device(cls, idx, eltype(idx))
    return (idx .== cls)
end

function collate_dense_tensors(samples::AbstractVector{<:AbstractArray}, pad_v::Real=0)
    isempty(samples) && return zeros(Float32, 0)
    ndims_set = unique(map(ndims, samples))
    length(ndims_set) != 1 && error("Samples has varying dimensions: $ndims_set")
    max_shape = map(maximum, zip(map(size, samples)...))
    first = samples[1]
    out = fill!(similar(first, eltype(first), length(samples), max_shape...), pad_v)
    for (i, t) in enumerate(samples)
        slices = ntuple(d -> 1:size(t, d), ndims(t))
        view(out, i, slices...) .= t
    end
    return out
end
