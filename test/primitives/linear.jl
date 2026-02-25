@testset "linear primitive" begin
    d_in, d_out, batch = 8, 12, 3
    x = randn(Float32, d_in, batch)
    W = randn(Float32, d_out, d_in)

    @testset "with bias" begin
        b = randn(Float32, d_out)
        y = Onion.linear(DefaultBackend(), x, W, b)
        @test size(y) == (d_out, batch)
        @test y ≈ W * x .+ b
    end

    @testset "without bias" begin
        y = Onion.linear(DefaultBackend(), x, W, false)
        @test size(y) == (d_out, batch)
        @test y ≈ W * x
    end

    @testset "top-level dispatch (no explicit backend)" begin
        b = randn(Float32, d_out)
        y_explicit = Onion.linear(DefaultBackend(), x, W, b)
        y_implicit = Onion.linear(x, W, b)
        @test y_explicit ≈ y_implicit
    end
end
