import Base: *
import LinearAlgebra

abstract type Postfix end
function postfix end
x * f::Postfix = postfix(f, x)

struct InversePostfix <: Postfix end
const ⁻¹ = InversePostfix()
postfix(::InversePostfix, x) = inv(x)

struct TransposePostfix <: Postfix end
const ᵀ = TransposePostfix()
postfix(::TransposePostfix, x) = transpose(x)
function postfix(::TransposePostfix, x::AbstractArray{<:Any,N}) where N
    perm = ntuple(i -> i < 3 ? 3 - i : i, N)
    PermutedDimsArray(x, perm)
end

struct HermitianPostfix <: Postfix end
const ᴴ = HermitianPostfix()
postfix(::HermitianPostfix, x) = LinearAlgebra.hermitian(x)

# (x)f(y) -> ((x)f)y
struct FixPostfix{P<:Postfix,T} <: Postfix
    f::P
    y::T
end
(f::Postfix)(x) = FixPostfix(f, x)
postfix((; f, y)::FixPostfix, x) = ((x)f)y
