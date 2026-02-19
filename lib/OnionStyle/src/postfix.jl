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

struct LiteralPostfix{T} <: Postfix end
const i8  = LiteralPostfix{Int8}()
const i16 = LiteralPostfix{Int16}()
const i32 = LiteralPostfix{Int32}()
const u8  = LiteralPostfix{UInt8}()
const u16 = LiteralPostfix{UInt16}()
const u32 = LiteralPostfix{UInt32}()
postfix(::LiteralPostfix{T}, x::Number) where T = T(x)