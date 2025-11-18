@testset "layers" begin

    include("BlockLinear.jl")
    include("Modulator.jl")
    include("DyT.jl")

    include("embedding/embedding.jl")
    include("norm/norm.jl")
    include("transformers/transformers.jl")

end