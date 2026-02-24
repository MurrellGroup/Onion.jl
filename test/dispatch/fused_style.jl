using SpecialFunctions: erf

@testset "FusedStyle dispatch" begin
    @testset "LayerStyle returns FusedStyle for fused layers" begin
        @test Onion.LayerStyle(DyT) isa Onion.FusedStyle
        @test Onion.LayerStyle(Derf) isa Onion.FusedStyle
        @test Onion.LayerStyle(AdaAffine) isa Onion.FusedStyle
    end

    @testset "LayerStyle returns EagerStyle for eager layers" begin
        @test Onion.LayerStyle(Linear) isa Onion.EagerStyle
        @test Onion.LayerStyle(RMSNorm) isa Onion.EagerStyle
    end

    @testset "DyT forward produces correct result" begin
        dim = 16
        dyt = DyT(dim)
        x = randn(Float32, dim, 4)
        y = dyt(x)
        @test size(y) == size(x)
        # Manual formula: weight * tanh(alpha * x) + bias
        ref = dyt.weight .* tanh.(dyt.alpha .* x) .+ dyt.bias
        @test y ≈ ref
    end

    @testset "Derf forward produces correct result" begin
        dim = 16
        derf = Derf(dim)
        x = randn(Float32, dim, 4)
        y = derf(x)
        @test size(y) == size(x)
        # Manual formula: γ * erf(α * x + s) + β
        ref = derf.γ .* erf.(derf.α .* x .+ derf.s) .+ derf.β
        @test y ≈ ref
    end

    @testset "RMSNorm has fuse method" begin
        dim = 16
        norm = RMSNorm(dim)
        x = randn(Float32, dim, 4)
        # Normal call should work
        y = norm(x)
        @test size(y) == size(x)
    end

    @testset "AdaAffine with FusedStyle inner layer" begin
        dim, cond_dim = 16, 8
        aa = AdaAffine(DyT(dim), dim, cond_dim)
        x = randn(Float32, dim, 4)
        cond = randn(Float32, cond_dim, 4)
        y = aa(x, cond)
        @test size(y) == (dim, 4)
    end
end
