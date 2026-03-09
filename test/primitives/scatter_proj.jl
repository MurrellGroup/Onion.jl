@testset "scatter_proj primitive" begin
    d_model, d_head, n_heads, k, L = 32, 8, 16, 4, 10
    w = randn(Float32, d_head, d_model, n_heads)
    x = randn(Float32, d_model, L)

    r = randn(Float32, n_heads, d_model)
    scores = r * x
    i = mapslices(v -> partialsortperm(v, 1:k, rev=true), scores, dims=1)

    @testset "correctness" begin
        y = Onion.scatter_proj(DefaultBackend(), w, x, i)
        @test size(y) == (d_head, k, L)

        ref = zeros(Float32, d_head, k, L)
        for l in 1:L, j in 1:k
            ref[:, j, l] = w[:, :, i[j, l]] * x[:, l]
        end
        @test y ≈ ref
    end

    @testset "implicit backend dispatch" begin
        y_explicit = Onion.scatter_proj(DefaultBackend(), w, x, i)
        y_implicit = Onion.scatter_proj(w, x, i)
        @test y_explicit ≈ y_implicit
    end
end
