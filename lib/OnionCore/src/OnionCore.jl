module OnionCore

include("Rules.jl")
export Rules

include("Backend.jl")
public Backend

include("Primitive.jl")
public Primitive
export @primitive

include("Layer.jl")
export Layer
public LayerStyle, apply, apply_with
public EagerStyle, forward

end
