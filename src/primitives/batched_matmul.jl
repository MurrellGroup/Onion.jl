@primitive batched_matmul
@primitive batched_matmul!

get_buffers(::typeof(batched_matmul), b::Backend, A, B) =
    (; C = similar(A, size(A, 1), size(B, 2), size(A, 3)))

function batched_matmul(b::Backend, A::AbstractArray{T,3}, B::AbstractArray{T,3}) where T
    bufs = get_buffers(batched_matmul, b, A, B)
    batched_matmul!(b, bufs.C, A, B)
    return bufs.C
end

function batched_matmul(::DefaultBackend, A::AbstractArray{T,3}, B::AbstractArray{T,3}) where T
    return NNlib.batched_mul(A, B)
end
