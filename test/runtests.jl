using Onion
using Test
using Flux

const ONION_TEST_UNET = get(ENV, "ONION_TEST_UNET", "false") == "true"

@testset "Onion.jl" verbose=true begin

    include("layers/layers.jl")
    
    if ONION_TEST_UNET
        include("UNet/test_unet.jl")
    else
        @info "Skipping UNet tests"
    end

end
