@testset "combine_projections primitive" begin
    C, L, B = 4, 6, 2
    a = randn(Float32, C, L, L, B)
    b = randn(Float32, C, L, L, B)

    @testset "outgoing: output shape" begin
        y = Onion.combine_projections(DefaultBackend(), a, b, true)
        @test size(y) == (C, L, L, B)
    end

    @testset "incoming: output shape" begin
        y = Onion.combine_projections(DefaultBackend(), a, b, false)
        @test size(y) == (C, L, L, B)
    end

    @testset "outgoing vs incoming differ" begin
        y_out = Onion.combine_projections(DefaultBackend(), a, b, true)
        y_in  = Onion.combine_projections(DefaultBackend(), a, b, false)
        @test !(y_out ≈ y_in)
    end

    @testset "top-level dispatch" begin
        y_explicit = Onion.combine_projections(DefaultBackend(), a, b, true)
        y_implicit = Onion.combine_projections(a, b, true)
        @test y_explicit ≈ y_implicit
    end
end
