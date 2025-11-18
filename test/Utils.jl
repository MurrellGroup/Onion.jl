@testset "Utils" begin

    @testset "like" begin
        v = rand(Int, 16)
        @test all(x -> x === 1.0f0, like(1.0f0, v, Float32, 2, 1))
        @test all(x -> x === 1.0f0, like(1, v, Float32, 4))
        @test all(x -> x === 1, like(1f0, v, 4))
        @test all(x -> x === 1, like(1f0, v))

        @test all(x -> x === 0f0, zeros_like(v, Float32, 20))
        @test all(x -> x === 1f0, ones_like(v, Float32, 20))
        @test all(x -> x === false, falses_like(v, 20))
        @test all(x -> x === true, trues_like(v, 20))

        @test let r = like(1:5, v)
            r == 1:5 && r isa Vector{Int}
        end
        @test let r = like(1:5, v, Float32)
            r == 1:5 && r isa Vector{Float32}
        end
        @test let r = like(1.0:5.0, v)
            r == 1:5 && r isa Vector{Float64}
        end
    end

    @testset "watmul" begin
        W = rand(Float32, 3, 2, 5)
        x = rand(Float32, 10, 7)
        @test watmul(W, x) == W ⨝ x
        @test size(watmul(W, x)) == (15, 7)
    end

end
