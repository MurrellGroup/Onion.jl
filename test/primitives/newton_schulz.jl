@testset "newton_schulz primitive" begin
    coeffs = fill((3.4445f0, -4.7750f0, 2.0315f0), 5)

    @testset "2D tall" begin
        X = randn(Float32, 16, 8) * 0.1f0
        Y = Onion.newton_schulz(DefaultBackend(), X, coeffs)
        @test size(Y) == (16, 8)
        @test all(isfinite.(Y))
    end

    @testset "2D wide" begin
        X = randn(Float32, 8, 16) * 0.1f0
        Y = Onion.newton_schulz(DefaultBackend(), X, coeffs)
        @test size(Y) == (8, 16)
        @test all(isfinite.(Y))
    end

    @testset "2D square" begin
        X = randn(Float32, 8, 8) * 0.1f0
        Y = Onion.newton_schulz(DefaultBackend(), X, coeffs)
        @test size(Y) == (8, 8)
    end

    @testset "3D batched" begin
        X = randn(Float32, 16, 8, 3) * 0.1f0
        Y = Onion.newton_schulz(DefaultBackend(), X, coeffs)
        @test size(Y) == (16, 8, 3)
    end

    @testset "singular value clustering" begin
        # Muon coefficients cluster σ into a narrow band near 1
        X = randn(Float32, 32, 16)
        X = X / norm(X)
        Y = Onion.newton_schulz(DefaultBackend(), X, coeffs)
        svals = svdvals(Y)
        @test all(0.5f0 .< svals .< 1.3f0)
        # variance should be much smaller than initial
        @test std(svals) < 0.2f0
    end

    @testset "reference tall" begin
        X = randn(Float32, 12, 6) * 0.1f0
        function ns_ref(X, coefficients)
            for (a, b, c) in coefficients
                A = X' * X
                X = a * X + X * (b * A + c * A * A)
            end
            return X
        end
        @test Onion.newton_schulz(DefaultBackend(), X, coeffs) ≈ ns_ref(X, coeffs) rtol=1f-5
    end

    @testset "reference wide" begin
        X = randn(Float32, 6, 12) * 0.1f0
        function ns_ref_wide(X, coefficients)
            for (a, b, c) in coefficients
                A = X * X'
                X = a * X + (b * A + c * A * A) * X
            end
            return X
        end
        @test Onion.newton_schulz(DefaultBackend(), X, coeffs) ≈ ns_ref_wide(X, coeffs) rtol=1f-5
    end

    @testset "3D matches per-batch 2D" begin
        X3 = randn(Float32, 12, 6, 3) * 0.1f0
        Y3 = Onion.newton_schulz(DefaultBackend(), X3, coeffs)
        for i in 1:3
            Y2 = Onion.newton_schulz(DefaultBackend(), X3[:, :, i], coeffs)
            @test Y3[:, :, i] ≈ Y2 rtol=1f-5
        end
    end

    @testset "top-level dispatch" begin
        X = randn(Float32, 12, 8) * 0.1f0
        y_explicit = Onion.newton_schulz(DefaultBackend(), X, coeffs)
        y_implicit = Onion.newton_schulz(X, coeffs)
        @test y_explicit ≈ y_implicit
    end
end
