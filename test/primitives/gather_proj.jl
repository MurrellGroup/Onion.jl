@testset "gather_proj primitive" begin
    d_model, d_head, n_heads, k, L = 32, 8, 16, 4, 10
    w = randn(Float32, d_model, d_head, n_heads)
    z = randn(Float32, d_head, k, L)
    i = rand(1:n_heads, k, L)

    @testset "correctness" begin
        o = Onion.gather_proj(DefaultBackend(), w, z, i)
        @test size(o) == (d_model, L)

        ref = zeros(Float32, d_model, L)
        for l in 1:L, j in 1:k
            ref[:, l] += w[:, :, i[j, l]] * z[:, j, l]
        end
        @test o ≈ ref
    end

    @testset "implicit backend dispatch" begin
        o_explicit = Onion.gather_proj(DefaultBackend(), w, z, i)
        o_implicit = Onion.gather_proj(w, z, i)
        @test o_explicit ≈ o_implicit
    end

    @testset "scatter_proj → gather_proj roundtrip" begin
        d, h, k_, l = 64, 4, 3, 7
        d_h = d ÷ h
        w_in = randn(Float32, d_h, d, h)
        w_out = randn(Float32, d, d_h, h)
        x = randn(Float32, d, l)
        i_ = rand(1:h, k_, l)

        y = Onion.scatter_proj(w_in, x, i_)
        @test size(y) == (d_h, k_, l)

        o = Onion.gather_proj(w_out, y, i_)
        @test size(o) == (d, l)
    end
end
