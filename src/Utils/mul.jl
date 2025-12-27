using MacroTools: @capture
using LinearAlgebra: Transpose
using NNlib: BatchedTranspose, batched_transpose, batched_mul, batched_mul!
using Einops

struct HighDimTranspose{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    x::A
end
Base.size(x::HighDimTranspose) = (size(x.x, 2), size(x.x, 1), size(x.x)[3:end]...)

ᵀ(X::Union{Number,AbstractArray{<:Any,0}}) = X
ᵀ(X::AbstractVecOrMat) = transpose(X)
ᵀ(X::AbstractArray{<:Any,3}) = batched_transpose(X)
ᵀ(X::AbstractArray) = HighDimTranspose(X)
Base.:*(X, ::typeof(ᵀ)) = ᵀ(X)

function _mul!(
    C::AbstractMatrix,
    A::AbstractVecOrMat,
    B::AbstractVecOrMat,
    args...
)
    mul!(C, A, B, args...)
    return C
end

function _mul!(
    C::AbstractArray{<:Any,3},
    A::AbstractArray{<:Any,3},
    B::AbstractArray{<:Any,3},
    args...
)
    batched_mul!(C, A, B, args...)
    return C
end

make_3d(x) = rearrange(x, einops"a b ... -> a b (...)")
make_3d(x::Transpose) = batched_transpose(make_3d(parent(x)))
make_3d(x::BatchedTranspose) = x
make_3d(x::HighDimTranspose) = batched_transpose(rearrange(x.x, einops"a b ... -> a b (...)"))

function _mul!(
    C::AbstractArray,
    A::Union{AbstractArray,HighDimTranspose},
    B::Union{AbstractArray,HighDimTranspose},
    args...
)
    ndims(C) == max(ndims(A), ndims(B)) ||
        throw(DimensionMismatch("C has $(ndims(C)) dimensions, but A has $(ndims(A)) and B has $(ndims(B)) dimensions"))
    C₃, A₃, B₃ = make_3d.((C, A, B))
    batched_mul!(C₃, A₃, B₃, args...)
    return C
end

macro mul!(ex)
    return :($_mul!($(esc.(_get_mul_args(ex))...)))
end

function _get_mul_args(ex)
    if @capture(ex, left_ += right_)
        b, C = _get_left_args(left)
        a, A, B = _get_right_args(right)
        C, A, B, a, b
    elseif @capture(ex, C_ = A_ * B_) || @capture(ex, C_ = (A_)^2)
        C, A, B
    else
        error("Expected left and right sides to be separated by + or =")
    end
end

function _get_left_args(ex)
    if @capture(ex, b_ * C_) || @capture(ex, b_(C_))
        b, C
    elseif @capture(ex, C_)
        1, C
    else
        error("Expected left side to be of the form b * C or C")
    end
end

function _get_right_args(ex)
    if @capture(ex, a_ * A_^2) || @capture(ex, a_(A_^2))
        a, A, A
    elseif @capture(ex, A_^2)
        1, A, A
    elseif @capture(ex, a_ * A_ * B_) || @capture(ex, a_(A_ * B_)) || @capture(ex, a_Number * (A_ * B_))
        a, A, B
    elseif @capture(ex, A_ * B_)
        1, A, B
    else
        error("Expected right side to be of the form a * A * B or A * B")
    end
end

mul(A, B) = batched_mul(A, B)

function mul(
    A::HighDimTranspose{<:Any,N},
    B::AbstractArray{<:Any,N}
) where N
    batch_size = size(A)[3:end]
    @assert batch_size == size(B)[3:end] "batch size has to be the same for the two arrays."
    C = batched_mul(make_3d(A), make_3d(B))
    return reshape(C, size(A, 1), size(B, 2), batch_size...)
end

function mul(
    A::AbstractArray{<:Any,N},
    B::HighDimTranspose{<:Any,N}
) where N
    batch_size = size(A)[3:end]
    @assert batch_size == size(B)[3:end] "batch size has to be the same for the two arrays."
    C = batched_mul(make_3d(A), make_3d(B))
    return reshape(C, size(A, 1), size(B, 2), batch_size...)
end

function mul(A::Transpose, B::AbstractArray)
    C = mul(make_3d(A), make_3d(B))
    return reshape(C, size(A, 1), size(B, 2), size(B)[3:end]...)
end

function mul(A::AbstractArray, B::Transpose)
    C = mul(make_3d(A), make_3d(B))
    return reshape(C, size(A, 1), size(B, 2), size(A)[3:end]...)
end
