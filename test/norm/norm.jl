@testset "norm.jl" begin
    @testset "PNorm" begin
        v = randn(Float32, 10)
        @test sum(abs, PNorm(1)(v)) ≈ 1f0
        @test PNorm(1, dims=())(v) == v
        @test sum(abs2, PNorm(2)(v)) ≈ 1f0
        @test PNorm(2, dims=())(v) == v
    end
end