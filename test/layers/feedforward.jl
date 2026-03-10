using NNlib: relu

@testset "StarGLU layer" begin
    @testset "basic forward" begin
        ff = StarGLU(8, 16)
        x = randn(Float32, 8, 5)
        y = ff(x)
        @test size(y) == (8, 5)
    end

    @testset "pair syntax" begin
        ff = StarGLU(8 => 16)
        x = randn(Float32, 8, 5)
        y = ff(x)
        @test size(y) == (8, 5)
    end

    @testset "3-arg (different output dim)" begin
        ff = StarGLU(8, 16, 12)
        x = randn(Float32, 8, 5)
        y = ff(x)
        @test size(y) == (12, 5)
    end

    @testset "custom activation" begin
        ff = StarGLU(8, 16; act=relu)
        x = randn(Float32, 8, 5)
        y = ff(x)
        @test size(y) == (8, 5)
    end

    @testset "3D input" begin
        ff = StarGLU(8, 16)
        x = randn(Float32, 8, 5, 3)
        y = ff(x)
        @test size(y) == (8, 5, 3)
    end
end
