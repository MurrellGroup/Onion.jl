@testset "RoPE layer" begin
    @testset "construction" begin
        rope = RoPE(16, 100)
        @test size(rope.cos, 1) == 8   # dim÷2
        @test size(rope.cos, 2) == 100
    end

    @testset "indexing" begin
        rope = RoPE(16, 100)
        sub = rope[1:10]
        @test size(sub.cos, 2) == 10
        @test size(sub.sin, 2) == 10
    end

    @testset "forward" begin
        rope = RoPE(16, 100)
        x = randn(Float32, 16, 10)
        y = rope[1:10](x)
        @test size(y) == (16, 10)
    end

    @testset "identity when cos=1, sin=0" begin
        # Construct a RoPE where all positions are 0 → cos=1, sin=0
        rope = RoPE(16, 1; start_pos=0)
        x = randn(Float32, 16, 1)
        y = rope(x)
        @test y ≈ x
    end

    @testset "4D input with heads and batch" begin
        rope = RoPE(16, 20)
        sub = rope[1:10]
        x = randn(Float32, 16, 10, 4, 2)
        y = sub(x)
        @test size(y) == (16, 10, 4, 2)
    end
end
