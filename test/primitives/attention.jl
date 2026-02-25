@testset "attention primitive" begin
    head_dim, seq_len, n_heads, batch = 8, 4, 2, 2

    q = randn(Float32, head_dim, seq_len, n_heads, batch)
    k = randn(Float32, head_dim, seq_len, n_heads, batch)
    v = randn(Float32, head_dim, seq_len, n_heads, batch)

    @testset "output shape" begin
        y = Onion.attention(DefaultBackend(), q, k, v)
        @test size(y) == (head_dim, seq_len, n_heads, batch)
    end

    @testset "self-attention (q=k=v)" begin
        y = Onion.attention(DefaultBackend(), q, q, q)
        @test size(y) == size(q)
        @test all(isfinite.(y))
    end

    @testset "causal masking" begin
        y = Onion.attention(DefaultBackend(), q, k, v; causal=true)
        @test size(y) == (head_dim, seq_len, n_heads, batch)
        @test all(isfinite.(y))
    end

    @testset "with pair bias" begin
        pair = randn(Float32, seq_len, seq_len, n_heads, batch)
        y = Onion.attention(DefaultBackend(), q, k, v; pair=pair)
        @test size(y) == (head_dim, seq_len, n_heads, batch)
    end

    @testset "GQA (fewer kv heads)" begin
        n_kv_heads = 1
        k_gqa = randn(Float32, head_dim, seq_len, n_kv_heads, batch)
        v_gqa = randn(Float32, head_dim, seq_len, n_kv_heads, batch)
        y = Onion.attention(DefaultBackend(), q, k_gqa, v_gqa)
        @test size(y) == (head_dim, seq_len, n_heads, batch)
    end

    @testset "top-level dispatch" begin
        y_explicit = Onion.attention(DefaultBackend(), q, k, v)
        y_implicit = Onion.attention(q, k, v)
        @test y_explicit ≈ y_implicit
    end
end
