module OnionStyle

include("postfix.jl")
export ⁻¹, ᵀ, ᴴ
public i8, i16, i32, u8, u16, u32

include("to.jl")
export →

# TODO: something superior to @concrete?
# - kwdef
# - no T(args...) constructor, but:
#   construct(T; kws...)

const Optional{T} = Union{T,Nothing}
export Optional

include("staticmap.jl")
export @staticmap

include("asmatrix.jl")
export @asmatrix

end
