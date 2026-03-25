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
        y = Onion.linear(DefaultBackend(), x, W, nothing)
        @test size(y) == (d_out, batch)
        @test y ≈ W * x
    end

    @testset "BackendMap dispatch" begin
        b = randn(Float32, d_out)
        mb = Onion.BackendMap(_ -> DefaultBackend())
        y_explicit = Onion.linear(DefaultBackend(), x, W, b)
        y_multi = Onion.linear(mb, x, W, b)
        @test y_explicit ≈ y_multi
    end
end
