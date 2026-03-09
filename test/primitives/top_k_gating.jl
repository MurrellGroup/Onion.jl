@testset "top_k_gating primitive" begin
    H, L = 8, 10
    k = 4
    scores = randn(Float32, H, L)

    @testset "correctness against partialsortperm" begin
        indices, values = Onion.top_k_gating(DefaultBackend(), scores, k)
        @test size(indices) == (k, L)
        @test size(values) == (k, L)

        for l in 1:L
            ref_perm = partialsortperm(view(scores, :, l), 1:k, rev=true)
            @test indices[:, l] == ref_perm
            @test values[:, l] ≈ scores[ref_perm, l]
        end
    end

    @testset "implicit backend dispatch" begin
        r1 = Onion.top_k_gating(DefaultBackend(), scores, k)
        r2 = Onion.top_k_gating(scores, k)
        @test r1[1] == r2[1]
        @test r1[2] ≈ r2[2]
    end

    @testset "k=1 (single head)" begin
        indices, values = Onion.top_k_gating(scores, 1)
        @test size(indices) == (1, L)
        @test size(values) == (1, L)
        for l in 1:L
            @test indices[1, l] == argmax(view(scores, :, l))
            @test values[1, l] ≈ maximum(view(scores, :, l))
        end
    end

    @testset "k=H (all heads)" begin
        indices, values = Onion.top_k_gating(scores, H)
        @test size(indices) == (H, L)
        @test size(values) == (H, L)
        # All scores should be present (just reordered)
        for l in 1:L
            @test sort(values[:, l]) ≈ sort(scores[:, l])
        end
    end

    @testset "indices are in 1:H" begin
        indices, _ = Onion.top_k_gating(scores, k)
        @test all(1 .<= indices .<= H)
    end

    @testset "indices are unique per column" begin
        indices, _ = Onion.top_k_gating(scores, k)
        for l in 1:L
            @test length(unique(indices[:, l])) == k
        end
    end
end
