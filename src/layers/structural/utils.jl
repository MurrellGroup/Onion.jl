# ──── OpenFold helper functions ────

function permute_final_dims(x::AbstractArray, order::NTuple{N,Int}) where {N}
    n = ndims(x)
    k = length(order)
    prefix = collect(1:(n - k))
    # order is 0-based, match OpenFold semantics
    tail = [n - k + o + 1 for o in order]
    return permutedims(x, vcat(prefix, tail))
end

function flatten_final_dims(x::AbstractArray, k::Int)
    n = ndims(x)
    new_shape = (size(x)[1:(n - k)]..., :)
    return reshape(x, new_shape)
end

function dict_multimap(fn, dicts::AbstractVector{<:AbstractDict})
    first_d = dicts[1]
    out = Dict{Symbol,Any}()
    for (k, v) in first_d
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

function one_hot_last(idx::AbstractArray{<:Integer}, num_classes::Int)
    cls = reshape(0:(num_classes - 1), ntuple(_ -> 1, ndims(idx))..., num_classes)
    return (idx .== cls)
end

function collate_dense_tensors(samples::AbstractVector{<:AbstractArray}, pad_v::Real=0)
    isempty(samples) && return zeros(Float32, 0)
    max_shape = map(maximum, zip(map(size, samples)...))
    first_s = samples[1]
    out = fill!(similar(first_s, eltype(first_s), length(samples), max_shape...), pad_v)
    for (i, t) in enumerate(samples)
        slices = ntuple(d -> 1:size(t, d), ndims(t))
        view(out, i, slices...) .= t
    end
    return out
end

# Device transfer utility
function to_device(x::AbstractArray, like::AbstractArray, ::Type{T}=eltype(x)) where {T}
    return @ignore_derivatives begin
        y = similar(like, T, size(x))
        copyto!(y, T.(x))
        y
    end
end

function to_device(x::Number, ::AbstractArray, ::Type{T}=typeof(x)) where {T}
    return @ignore_derivatives T(x)
end
