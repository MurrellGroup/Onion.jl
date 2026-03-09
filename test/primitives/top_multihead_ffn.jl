@testset "top_multihead_ffn primitive" begin
    D, I, H, k, L = 16, 32, 8, 4, 10

    K = randn(Float32, I, D, H)
    U = randn(Float32, I, D, H)
    V = randn(Float32, D, I, H)

    # Build deterministic routing indices
    i = rand(1:H, k, L)

    # Input Q from a mock scatter_proj
    Q = randn(Float32, D, k, L)

    @testset "correctness against manual loop" begin
        y = Onion.top_multihead_ffn(DefaultBackend(), Q, K, U, V, Onion.swish, i)
        @test size(y) == (D, k, L)

        ref = zeros(Float32, D, k, L)
        for l in 1:L, j in 1:k
            h = i[j, l]
            q = Q[:, j, l]
            ref[:, j, l] = V[:, :, h] * (Onion.swish.(K[:, :, h] * q) .* (U[:, :, h] * q))
        end
        @test y ≈ ref
    end

    @testset "implicit backend dispatch" begin
        y_explicit = Onion.top_multihead_ffn(DefaultBackend(), Q, K, U, V, Onion.swish, i)
        y_implicit = Onion.top_multihead_ffn(Q, K, U, V, Onion.swish, i)
        @test y_explicit ≈ y_implicit
    end

    @testset "k=1 (single head per token)" begin
        i1 = rand(1:H, 1, L)
        Q1 = randn(Float32, D, 1, L)
        y1 = Onion.top_multihead_ffn(Q1, K, U, V, Onion.swish, i1)
        @test size(y1) == (D, 1, L)
    end

    @testset "k=H (all heads)" begin
        i_all = repeat(1:H, 1, L)
        Q_all = randn(Float32, D, H, L)
        y_all = Onion.top_multihead_ffn(Q_all, K, U, V, Onion.swish, i_all)
        @test size(y_all) == (D, H, L)
    end

    @testset "end-to-end: scatter_proj → top_multihead_ffn → gather_proj" begin
        d_model, d_h, n_heads = 64, 16, 8
        k_, l_ = 4, 7
        I_ = 32

        w_in = randn(Float32, d_h, d_model, n_heads)
        K_ = randn(Float32, I_, d_h, n_heads)
        U_ = randn(Float32, I_, d_h, n_heads)
        V_ = randn(Float32, d_h, I_, n_heads)
        w_out = randn(Float32, d_model, d_h, n_heads)

        x = randn(Float32, d_model, l_)
        r = randn(Float32, n_heads, d_model)
        scores = r * x
        i_ = mapslices(v -> partialsortperm(v, 1:k_, rev=true), scores, dims=1)

        # Input projection (fan-out)
        q = Onion.scatter_proj(w_in, x, i_)
        @test size(q) == (d_h, k_, l_)

        # Sparse MHFFN
        s = Onion.top_multihead_ffn(q, K_, U_, V_, Onion.swish, i_)
        @test size(s) == (d_h, k_, l_)

        # Output projection (fan-in with fused k-reduction)
        out = Onion.gather_proj(w_out, s, i_)
        @test size(out) == (d_model, l_)
    end
end
