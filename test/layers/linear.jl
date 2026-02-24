@testset "Linear layer" begin
    @testset "basic forward" begin
        l = Linear(8 => 12)
        x = randn(Float32, 8, 5)
        y = l(x)
        @test size(y) == (12, 5)
    end

    @testset "without bias" begin
        l = Linear(8 => 12, bias=false)
        x = randn(Float32, 8, 5)
        y = l(x)
        @test size(y) == (12, 5)
        @test l.bias === false
    end

    @testset "with bias" begin
        l = Linear(8 => 12, bias=true)
        x = randn(Float32, 8, 5)
        y = l(x)
        @test size(y) == (12, 5)
        @test l.bias isa AbstractVector
        @test length(l.bias) == 12
    end

    @testset "3D input reshaping" begin
        l = Linear(8 => 12)
        x = randn(Float32, 8, 5, 3)
        y = l(x)
        @test size(y) == (12, 5, 3)
    end

    @testset "weight dimensions" begin
        l = Linear(8 => 12)
        @test size(l.weight) == (12, 8)
    end
end
