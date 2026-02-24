@testset "Attention layer" begin
    @testset "self-attention basic" begin
        dim, n_heads = 32, 4
        attn = Attention(dim, n_heads)
        x = randn(Float32, dim, 8, 2)
        y = attn(x)
        @test size(y) == (dim, 8, 2)
    end

    @testset "cross-attention" begin
        dim, n_heads = 32, 4
        attn = Attention(dim, n_heads)
        q = randn(Float32, dim, 8, 2)
        k = randn(Float32, dim, 6, 2)
        y = attn(q, k)
        @test size(y) == (dim, 8, 2)
    end

    @testset "GQA (n_kv_heads < n_heads)" begin
        dim, n_heads, n_kv_heads = 32, 4, 2
        attn = Attention(dim, n_heads, n_kv_heads)
        x = randn(Float32, dim, 8, 2)
        y = attn(x)
        @test size(y) == (dim, 8, 2)
    end

    @testset "causal attention" begin
        dim, n_heads = 32, 4
        attn = Attention(dim, n_heads)
        x = randn(Float32, dim, 8, 2)
        y = attn(x; causal=true)
        @test size(y) == (dim, 8, 2)
    end

    @testset "qk_norm" begin
        dim, n_heads = 32, 4
        attn = Attention(dim, n_heads; qk_norm=true)
        x = randn(Float32, dim, 8, 2)
        y = attn(x)
        @test size(y) == (dim, 8, 2)
    end

    @testset "in_dim field is correct after fix" begin
        dim, n_heads = 64, 8
        attn = Attention(dim, n_heads)
        @test attn.in_dim == dim
        @test attn.head_dim == dim ÷ n_heads
    end

    @testset "custom head_dim" begin
        dim, n_heads, head_dim = 32, 4, 16
        attn = Attention(dim, n_heads; head_dim=head_dim)
        @test attn.head_dim == 16
        x = randn(Float32, dim, 8, 2)
        y = attn(x)
        @test size(y) == (dim, 8, 2)
    end
end

@testset "KVCache" begin
    @testset "construction via kv_cache" begin
        dim, n_heads = 32, 4
        attn = Attention(dim, n_heads)
        cache = kv_cache(attn, 64, 2)
        @test size(cache.k) == (dim ÷ n_heads, 64, n_heads, 2)
        @test pos(cache) == 0
    end

    @testset "pos and pos!" begin
        dim, n_heads = 32, 4
        attn = Attention(dim, n_heads)
        cache = kv_cache(attn, 64, 2)
        @test pos(cache) == 0
        pos!(cache, 10)
        @test pos(cache) == 10
    end

    @testset "pos! bounds check" begin
        dim, n_heads = 32, 4
        attn = Attention(dim, n_heads)
        cache = kv_cache(attn, 16, 1)
        @test_throws BoundsError pos!(cache, 17)
    end

    @testset "extend" begin
        dim, n_heads = 32, 4
        attn = Attention(dim, n_heads)
        cache = kv_cache(attn, 16, 1)
        cache2 = extend(cache, 32)
        @test size(cache2.k, 2) == 32
    end
end
