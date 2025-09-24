@testset "norm.jl" begin
    @testset "LpNorm" begin
        v = randn(Float32, 10)
        for p in 1:4
            @test sum(x -> abs(x)^p, LpNorm(p)(v)) ≈ 1f0
            @test all(x -> abs(x) ≈ 1f0, LpNorm(p, dims=())(v))
        end
    end
end
