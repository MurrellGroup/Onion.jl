module OnionStyle

include("postfix.jl")
export ⁻¹, ᵀ, ᴴ

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

end
