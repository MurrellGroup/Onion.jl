@testset "TransformerBlock" begin
    @testset "basic forward" begin
        dim, n_heads = 32, 4
        block = TransformerBlock(dim, n_heads)
        x = randn(Float32, dim, 8, 2)
        y = block(x)
        @test size(y) == (dim, 8, 2)
    end

    @testset "with GQA" begin
        dim, n_heads, n_kv_heads = 32, 4, 2
        block = TransformerBlock(dim, n_heads, n_kv_heads)
        x = randn(Float32, dim, 8, 2)
        y = block(x)
        @test size(y) == (dim, 8, 2)
    end

    @testset "with RoPE" begin
        dim, n_heads, seq_len = 32, 4, 8
        block = TransformerBlock(dim, n_heads)
        rope = RoPE(dim ÷ n_heads, 100)
        x = randn(Float32, dim, seq_len, 2)
        y = block(x; rope=rope[1:seq_len])
        @test size(y) == (dim, seq_len, 2)
    end

    @testset "with causal" begin
        dim, n_heads = 32, 4
        block = TransformerBlock(dim, n_heads)
        x = randn(Float32, dim, 8, 2)
        y = block(x; causal=true)
        @test size(y) == (dim, 8, 2)
    end

    @testset "custom ff_hidden_dim" begin
        dim, n_heads = 32, 4
        block = TransformerBlock(dim, n_heads, n_heads, 64)
        x = randn(Float32, dim, 8, 2)
        y = block(x)
        @test size(y) == (dim, 8, 2)
    end
end
