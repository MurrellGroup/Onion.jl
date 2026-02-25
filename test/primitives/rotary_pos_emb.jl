@testset "rotary_pos_emb primitive" begin
    dim, seq_len = 8, 4
    x = randn(Float32, dim, seq_len)

    @testset "output shape" begin
        cos_vals = ones(Float32, dim ÷ 2, seq_len)
        sin_vals = zeros(Float32, dim ÷ 2, seq_len)
        y = Onion.rotary_pos_emb(DefaultBackend(), x, cos_vals, sin_vals)
        @test size(y) == (dim, seq_len)
    end

    @testset "identity when cos=1, sin=0" begin
        cos_vals = ones(Float32, dim ÷ 2, seq_len)
        sin_vals = zeros(Float32, dim ÷ 2, seq_len)
        y = Onion.rotary_pos_emb(DefaultBackend(), x, cos_vals, sin_vals)
        @test y ≈ x
    end

    @testset "correctness (manual rotation)" begin
        cos_vals = randn(Float32, dim ÷ 2, seq_len)
        sin_vals = randn(Float32, dim ÷ 2, seq_len)
        y = Onion.rotary_pos_emb(DefaultBackend(), x, cos_vals, sin_vals)
        x1 = x[1:dim÷2, :]
        x2 = x[dim÷2+1:end, :]
        ref = vcat(
            x1 .* cos_vals .- x2 .* sin_vals,
            x2 .* cos_vals .+ x1 .* sin_vals,
        )
        @test y ≈ ref
    end

    @testset "4D input (head_dim, seq, heads, batch)" begin
        n_heads, batch = 2, 2
        x4 = randn(Float32, dim, seq_len, n_heads, batch)
        cos_vals = ones(Float32, dim ÷ 2, seq_len, 1, 1)
        sin_vals = zeros(Float32, dim ÷ 2, seq_len, 1, 1)
        y = Onion.rotary_pos_emb(DefaultBackend(), x4, cos_vals, sin_vals)
        @test size(y) == (dim, seq_len, n_heads, batch)
        @test y ≈ x4
    end

    @testset "top-level dispatch" begin
        cos_vals = randn(Float32, dim ÷ 2, seq_len)
        sin_vals = randn(Float32, dim ÷ 2, seq_len)
        y_explicit = Onion.rotary_pos_emb(DefaultBackend(), x, cos_vals, sin_vals)
        y_implicit = Onion.rotary_pos_emb(x, cos_vals, sin_vals)
        @test y_explicit ≈ y_implicit
    end
end
