using Onion
using Test
using Flux

const ONION_TEST_UNET = get(ENV, "ONION_TEST_UNET", "false") == "true"

@testset "Onion.jl" begin

    include("BlockDense.jl")
    include("Modulator.jl")
    include("DyT.jl")

    include("embedding/embedding.jl")
    include("norm/norm.jl")
    include("transformers/transformers.jl")
    
    if ONION_TEST_UNET
        include("UNet/UNet.jl")
    else
        @info "Skipping UNet tests"
    end

end
