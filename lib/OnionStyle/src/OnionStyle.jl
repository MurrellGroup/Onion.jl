module OnionStyle

include("postfix.jl")
export ⁻¹, ᵀ, ᴴ
public i8, i16, i32, u8, u16, u32

include("to.jl")
export →

const Optional{T} = Union{T,Nothing}
export Optional

include("staticmap.jl")
export @staticmap

# re-export ConcreteStructs.@concrete?

end
