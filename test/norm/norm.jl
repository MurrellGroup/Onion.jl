@testset "norm.jl" begin
    @testset "PNorm" begin
        v = randn(Float32, 10)
        for p in 1:4
            @test sum(x -> abs(x)^p, PNorm(p)(v)) ≈ 1f0
            @test all(x -> abs(x) ≈ 1f0, PNorm(p, dims=())(v))
        end
    end
end
