using Onion
using Test
using Flux

const ONION_TEST_UNET = get(ENV, "ONION_TEST_UNET", "false") == "true"

@testset "Onion.jl" begin

    include("BlockDense.jl")
    include("DyT.jl")

    include("transformers/transformers.jl")
    include("embedding/embedding.jl")
    
    if ONION_TEST_UNET
        include("UNet/UNet.jl")
    else
        @info "Skipping UNet tests"
    end

end
