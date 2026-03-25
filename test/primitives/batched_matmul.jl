@testset "batched_matmul" begin
    M, K, N, B = 8, 6, 4, 3
    A = randn(Float32, M, K, B)
    B_arr = randn(Float32, K, N, B)

    C = Onion.batched_matmul(DefaultBackend(), A, B_arr)
    @test size(C) == (M, N, B)
    @test all(isfinite, C)

    # Verify against manual per-batch matmul
    for b in 1:B
        @test C[:, :, b] ≈ A[:, :, b] * B_arr[:, :, b] atol=1e-5
    end

    @testset "DefaultBackend matches NNlib" begin
        using NNlib: batched_mul
        C_ref = batched_mul(A, B_arr)
        @test C ≈ C_ref atol=1e-5
    end
end
