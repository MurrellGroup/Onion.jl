using NNlib

# permute_final_dims and flatten_final_dims are already defined in
# protein/openfold_utils.jl — do not redefine here to avoid method overwrite.

function chunk_layer(layer, inputs::Dict{Symbol,<:AbstractArray}; chunk_size::Int, no_batch_dims::Int)
    if isempty(inputs)
        error("Must provide at least one input")
    end
    batch_dims_list = [size(v)[1:no_batch_dims] for v in values(inputs)]
    orig_batch = map(max, zip(batch_dims_list...)) |> Tuple
    flat_batch = prod(orig_batch)

    prepped = Dict{Symbol,AbstractArray}()
    for (k, v) in inputs
        reps_batch = ntuple(i -> orig_batch[i] ÷ size(v, i), no_batch_dims)
        reps_tail = ntuple(_ -> 1, ndims(v) - no_batch_dims)
        reps = (reps_batch..., reps_tail...)
        v_exp = all(reps .== 1) ? v : repeat(v, reps...)
        prepped[k] = reshape(v_exp, flat_batch, size(v_exp)[no_batch_dims+1:end]...)
    end

    out = nothing
    for start in 1:chunk_size:flat_batch
        stop = min(start + chunk_size - 1, flat_batch)
        chunk_inputs = Dict{Symbol,AbstractArray}()
        for (k, v) in prepped
            tail = ntuple(_ -> Colon(), ndims(v) - 1)
            chunk_inputs[k] = view(v, start:stop, tail...)
        end
        chunk_out = layer(; chunk_inputs...)
        if out === nothing
            out = similar(chunk_out, (flat_batch, size(chunk_out)[2:end]...))
        end
        tail_out = ntuple(_ -> Colon(), ndims(chunk_out) - 1)
        view(out, start:stop, tail_out...) .= chunk_out
    end

    return reshape(out, orig_batch..., size(out)[2:end]...)
end
