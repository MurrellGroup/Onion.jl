@testset "norm.jl" begin

    @testset "LpNorm" begin
        v = randn(Float32, 10, 20)
        @testset for p in 1:5
            rtol = 1f-2

            s_all = sum(x -> abs(x)^p, LpNorm(p, dims=(1,2))(v))
            @test isapprox(s_all, 1f0; rtol)

            s_dim1 = sum(x -> abs(x)^p, LpNorm(p, dims=1)(v), dims=1)
            @test all(x -> isapprox(x, 1f0; rtol), s_dim1)

            s_el = abs.(LpNorm(p, dims=())(v))
            @test all(x -> isapprox(x, 1f0; rtol), s_el)
        end
    end

end
