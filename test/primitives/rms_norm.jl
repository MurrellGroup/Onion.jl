@testset "rms_norm primitive" begin
    dim, batch = 16, 4
    x = randn(Float32, dim, batch)
    w = ones(Float32, dim)

    @testset "with weight (offset=0)" begin
        y = Onion.rms_norm(DefaultBackend(), x, w; eps=1f-5, offset=0f0)
        @test size(y) == (dim, batch)
        # RMS of each column should be ≈ 1
        rms = sqrt.(mean(y .^ 2; dims=1))
        @test all(isapprox.(rms, 1f0; atol=1f-3))
    end

    @testset "with weight (offset=1, zero_centered)" begin
        w_zero = zeros(Float32, dim)
        y = Onion.rms_norm(DefaultBackend(), x, w_zero; eps=1f-5, offset=1f0)
        @test size(y) == (dim, batch)
        # offset=1 means (w + 1) = 1, so same as unit weight
        rms = sqrt.(mean(y .^ 2; dims=1))
        @test all(isapprox.(rms, 1f0; atol=1f-3))
    end

    @testset "without weight" begin
        y = Onion.rms_norm(DefaultBackend(), x; eps=1f-5)
        @test size(y) == (dim, batch)
        rms = sqrt.(mean(y .^ 2; dims=1))
        @test all(isapprox.(rms, 1f0; atol=1f-3))
    end

    @testset "top-level dispatch" begin
        y_explicit = Onion.rms_norm(DefaultBackend(), x, w; eps=1f-5, offset=0f0)
        y_implicit = Onion.rms_norm(x, w; eps=1f-5, offset=0f0)
        @test y_explicit ≈ y_implicit
    end
end
