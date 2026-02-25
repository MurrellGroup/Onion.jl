@testset "layer_norm primitive" begin
    dim, batch = 16, 4
    x = randn(Float32, dim, batch)
    w = ones(Float32, dim)
    b = zeros(Float32, dim)

    @testset "output shape" begin
        y = Onion.layer_norm(DefaultBackend(), x, w, b; eps=1f-6)
        @test size(y) == (dim, batch)
    end

    @testset "zero mean, unit variance" begin
        y = Onion.layer_norm(DefaultBackend(), x, w, b; eps=1f-6)
        col_means = mean(y; dims=1)
        col_vars = var(y; dims=1, corrected=false)
        @test all(isapprox.(col_means, 0f0; atol=1f-5))
        @test all(isapprox.(col_vars, 1f0; atol=1f-3))
    end

    @testset "with non-trivial weight and bias" begin
        w2 = randn(Float32, dim) .+ 2f0
        b2 = randn(Float32, dim)
        y = Onion.layer_norm(DefaultBackend(), x, w2, b2; eps=1f-6)
        @test size(y) == (dim, batch)
    end

    @testset "top-level dispatch" begin
        y_explicit = Onion.layer_norm(DefaultBackend(), x, w, b; eps=1f-6)
        y_implicit = Onion.layer_norm(x, w, b; eps=1f-6)
        @test y_explicit ≈ y_implicit
    end
end
