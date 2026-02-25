@testset "softmax primitive" begin
    dim, batch = 16, 4
    x = randn(Float32, dim, batch)

    @testset "output shape" begin
        y = Onion.softmax(DefaultBackend(), x)
        @test size(y) == (dim, batch)
    end

    @testset "sums to 1 along dim=1" begin
        y = Onion.softmax(DefaultBackend(), x)
        col_sums = sum(y; dims=1)
        @test all(isapprox.(col_sums, 1f0; atol=1f-6))
    end

    @testset "all values in [0, 1]" begin
        y = Onion.softmax(DefaultBackend(), x)
        @test all(y .>= 0f0)
        @test all(y .<= 1f0)
    end

    @testset "custom dims" begin
        y = Onion.softmax(DefaultBackend(), x, 2)
        row_sums = sum(y; dims=2)
        @test all(isapprox.(row_sums, 1f0; atol=1f-6))
    end

    @testset "top-level dispatch" begin
        y_explicit = Onion.softmax(DefaultBackend(), x)
        y_implicit = Onion.softmax(x)
        @test y_explicit ≈ y_implicit
    end
end
