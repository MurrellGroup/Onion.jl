@testset "norm.jl" begin
    @testset "LpNorm" begin
        v = randn(Float32, 10, 20)
        for p in 1:5
            @test sum(x -> abs(x)^p, LpNorm(p, dims=(1,2))(v)) ≈ 1f0
            @test all(x -> abs(x) ≈ 1f0, sum(x -> abs(x)^p, LpNorm(p, dims=1)(v), dims=1))
            @test all(x -> abs(x) ≈ 1f0, LpNorm(p, dims=())(v))
        end
    end
end
