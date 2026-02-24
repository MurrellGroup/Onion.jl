@testset "Composability layers" begin
    @testset "Composed" begin
        inner = Linear(8 => 12)
        outer = Linear(12 => 6)
        c = Composed(inner, outer)
        x = randn(Float32, 8, 5)
        y = c(x)
        @test size(y) == (6, 5)
        @test y ≈ outer(inner(x))
    end

    @testset "With" begin
        wrapper = ResidualConnection()
        layer = Linear(8 => 8)
        w = With(wrapper, layer)
        x = randn(Float32, 8, 5)
        y = w(x)
        @test size(y) == (8, 5)
        @test y ≈ layer(x) + x
    end

    @testset "AdaAffine" begin
        dim, cond_dim = 16, 8
        # AdaAffine wraps a FusedStyle layer (e.g. RMSNorm, DyT)
        aa = AdaAffine(RMSNorm(dim), dim, cond_dim)
        x = randn(Float32, dim, 4)
        cond = randn(Float32, cond_dim, 4)
        y = aa(x, cond)
        @test size(y) == (dim, 4)
    end

    @testset "AdaAffine with init_zero" begin
        dim, cond_dim = 16, 8
        aa = AdaAffine(RMSNorm(dim), dim, cond_dim; init_zero=true)
        x = randn(Float32, dim, 4)
        cond = randn(Float32, cond_dim, 4)
        y = aa(x, cond)
        @test size(y) == (dim, 4)
    end

    @testset "Modulator" begin
        gate = Modulator(8 => 12)
        Y = randn(Float32, 12, 5)
        X = randn(Float32, 8, 5)
        y = gate(Y, X)
        @test size(y) == (12, 5)
    end
end
