using LinearAlgebra: mul!
using ChainRulesCore
using ForwardDiff: derivative

# applies:
#   down * (act.(gate * x) .* (up * x))
function _star_glu_chunk!(
    (; b1, b2, y),
    (; up, gate, down, act),
    x::AbstractMatrix
)
    mul!(b1, up, x)
    mul!(b2, gate, x)
    b1 .*= act.(b2)
    mul!(y, down, b1)
    return nothing
end

@views function _star_glu_chunks!(
    (; b1, b2, y),
    (; up, gate, down, act),
    x::AbstractMatrix;
    chunk_size::Int,
    n_chunks::Int,
    n_trailing::Int,
)
    for i in 1:n_chunks
        _star_glu_chunk!(
            (; b1, b2, y=y[:, (1:chunk_size) .+ (i-1)chunk_size]),
            (; up, gate, down, act),
            x[:, (1:chunk_size) .+ (i-1)chunk_size]
        )
    end
    if n_trailing > 0
        _star_glu_chunk!(
            (; b1 = b1[:, 1:n_trailing],
               b2 = b2[:, 1:n_trailing],
               y = y[:, (1:n_trailing) .+ n_chunks * chunk_size]),
            (; up, gate, down, act),
            x[:, (1:n_trailing) .+ n_chunks * chunk_size]
        )
    end
    return nothing
end

function _star_glu(
    (; up, gate, down, act), x::AbstractMatrix;
    chunk_size::Int
)
    width = size(x, 2)
    intermediate_size = size(up, 1)
    n_chunks, n_trailing = divrem(width, chunk_size)
    buffer_width = n_chunks > 0 ? chunk_size : n_trailing
    b1 = similar(x, intermediate_size, buffer_width)
    b2 = similar(x, intermediate_size, buffer_width)
    y  = similar(x, size(down,1), width)
    _star_glu_chunks!((; b1, b2, y), (; up, gate, down, act), x; chunk_size, n_chunks, n_trailing)
    return (; b1, b2, y)
end

function star_glu((; up, gate, down, act), x::AbstractArray; kws...)
    x′ = reshape(x, size(x, 1), :)
    y′ = _star_glu((; up, gate, down, act), x′; kws...).y
    return reshape(y′, size(down, 1), size(x)[2:end]...)
end

function ChainRulesCore.rrule(::typeof(star_glu),
    (; up, gate, down, act), x::AbstractArray;
    chunk_size::Int
)
    x_reshaped = reshape(x, size(x, 1), :)
    # The forward pass does not need to return b1 and b2
    (; b1, b2, y) = _star_glu((; up, gate, down, act), x_reshaped; chunk_size)
    y_reshaped = y

    @views function pullback(dy)
        dy = unthunk(dy)
        ∇y = reshape(dy, size(dy, 1), :)
        
        # Determine chunking strategy from inputs
        n_chunks, n_trailing = divrem(size(x_reshaped, 2), chunk_size)
        buffer_width = n_chunks > 0 ? chunk_size : n_trailing

        # Allocate gradient accumulators
        ∇x = similar(x_reshaped)
        ∇up = zero(up)
        ∇gate = zero(gate)
        ∇down = zero(down)

        # Allocate per-chunk buffers
        b2_act = similar(b1)
        b1_gated = similar(b1)
        b2_act_deriv = similar(b1)
        ∇b1_gated = similar(b1)
        ∇b1 = similar(b1)
        ∇b2 = similar(b1)

        # Process full chunks
        for i in 1:n_chunks
            x_chunk = x_reshaped[:, (1:chunk_size) .+ (i-1)chunk_size]
            ∇y_chunk = ∇y[:, (1:chunk_size) .+ (i-1)chunk_size]
            ∇x_chunk = ∇x[:, (1:chunk_size) .+ (i-1)chunk_size]

            # Re-compute forward pass for chunk
            mul!(b1, up, x_chunk)
            mul!(b2, gate, x_chunk)
            b2_act .= act.(b2)
            b1_gated .= b1 .* b2_act
            b2_act_deriv .= derivative.(act, b2)

            # Backward pass for chunk
            mul!(∇b1_gated, down', ∇y_chunk)
            mul!(∇down, ∇y_chunk, b1_gated', 1, 1)

            ∇b1 .= ∇b1_gated .* b2_act
            ∇b2 .= ∇b1_gated .* b1
            ∇b2 .*= b2_act_deriv

            mul!(∇up, ∇b1, x_chunk', 1, 1)
            mul!(∇gate, ∇b2, x_chunk', 1, 1)

            mul!(∇x_chunk, up', ∇b1)
            mul!(∇x_chunk, gate', ∇b2, 1, 1)
        end

        # Handle trailing chunk
        @inbounds if n_trailing > 0
            x_chunk = x_reshaped[:, (1:n_trailing) .+ n_chunks * chunk_size]
            ∇y_chunk = ∇y[:, (1:n_trailing) .+ n_chunks * chunk_size]
            ∇x_chunk = ∇x[:, (1:n_trailing) .+ n_chunks * chunk_size]

            # Create views into buffers for the trailing chunk
            b1_trail = b1[:, 1:n_trailing]
            b2_trail = b2[:, 1:n_trailing]
            b2_act_trail = b2_act[:, 1:n_trailing]
            b1_gated_trail = b1_gated[:, 1:n_trailing]
            b2_act_deriv_trail = b2_act_deriv[:, 1:n_trailing]
            ∇b1_gated_trail = ∇b1_gated[:, 1:n_trailing]
            ∇b1_trail = ∇b1[:, 1:n_trailing]
            ∇b2_trail = ∇b2[:, 1:n_trailing]

            # Re-compute forward pass for trailing chunk
            mul!(b1_trail, up, x_chunk)
            mul!(b2_trail, gate, x_chunk)
            b2_act_trail .= act.(b2_trail)
            b1_gated_trail .= b1_trail .* b2_act_trail
            b2_act_deriv_trail .= derivative.(act, b2_trail)

            # Backward pass for trailing chunk
            mul!(∇b1_gated_trail, down', ∇y_chunk)
            mul!(∇down, ∇y_chunk, b1_gated_trail', 1, 1)

            ∇b1_trail .= ∇b1_gated_trail .* b2_act_trail
            ∇b2_trail .= ∇b1_gated_trail .* b1_trail
            ∇b2_trail .*= b2_act_deriv_trail

            mul!(∇up, ∇b1_trail, x_chunk', 1, 1)
            mul!(∇gate, ∇b2_trail, x_chunk', 1, 1)

            mul!(∇x_chunk, up', ∇b1_trail)
            mul!(∇x_chunk, gate', ∇b2_trail, 1, 1)
        end

        ∇layer = (; up=∇up, gate=∇gate, down=∇down, act=NoTangent())
        return NoTangent(), ∇layer, reshape(∇x, size(x))
    end
    return reshape(y_reshaped, size(down, 1), size(x)[2:end]...), pullback
end
