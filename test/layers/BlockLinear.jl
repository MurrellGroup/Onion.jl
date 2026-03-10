@testset "BlockLinear.jl" begin

    @testset "Linear vs BlockLinear equivalence when k=1" begin
        in_dim = 5
        out_dim = 7

        block_linear = BlockLinear(in_dim => out_dim, 1)
        linear = Linear(in_dim => out_dim)
        linear.weight .= block_linear.weight

        x = rand(Float32, in_dim)
        @test linear(x) ≈ block_linear(x)
    end

    @testset "BlockLinear vs sparse Linear equivalence for any k" begin

        for k in 1:4
            s1, s2 = 5, 7

            in_dim = s1 * k
            out_dim = s2 * k

            linear = Linear(in_dim => out_dim)
            block_linear = BlockLinear(in_dim => out_dim, k)

            linear.weight .= false
            for i in 1:k
                linear.weight[(1:s2) .+ s2 * (i-1), (1:s1) .+ s1 * (i-1)] .= block_linear.weight[:,:,i]
            end

            x = rand(Float32, in_dim)
            @test linear(x) ≈ block_linear(x)
        end

    end

end
