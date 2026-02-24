@testset "Normalization layers" begin
    @testset "LayerNorm" begin
        ln = Onion.LayerNorm(16)
        x = randn(Float32, 16, 4)
        y = ln(x)
        @test size(y) == (16, 4)
        # Should have approximately zero mean and unit variance per column
        col_means = mean(y; dims=1)
        @test all(isapprox.(col_means, 0f0; atol=1f-5))
    end

    @testset "RMSNorm" begin
        norm = RMSNorm(16)
        x = randn(Float32, 16, 4)
        y = norm(x)
        @test size(y) == (16, 4)
    end

    @testset "RMSNorm zero_centered" begin
        norm = RMSNorm(16; zero_centered=true)
        @test norm.offset == 1f0
        x = randn(Float32, 16, 4)
        y = norm(x)
        @test size(y) == (16, 4)
    end

    @testset "AdaLN" begin
        aln = AdaLN(16, 8)
        x = randn(Float32, 16, 4, 1)
        cond = randn(Float32, 8, 1)
        y = aln(x, cond)
        @test size(y) == (16, 4, 1)
    end

    @testset "LpNorm layer" begin
        for p in [1, 2, 3]
            norm = LpNorm(p)
            x = randn(Float32, 16, 4)
            y = norm(x)
            @test size(y) == (16, 4)
        end
    end

    @testset "L2Norm alias" begin
        norm = L2Norm()
        x = randn(Float32, 16, 4)
        y = norm(x)
        @test size(y) == (16, 4)
    end
end
