@testset "BlockDense.jl" begin

    @testset "Dense vs BlockDense equivalence when k=1" begin
        in_dim = 5
        out_dim = 7

        block_dense = BlockDense(in_dim => out_dim, 1, sigmoid)
        dense = Dense(in_dim => out_dim, sigmoid)
        dense.weight .= block_dense.weight

        x = rand(Float32, in_dim)
        @test dense(x) ≈ block_dense(x)
    end

    @testset "BlockDense vs sparse Dense equivalence for any k" begin

        for k in 1:4
            s1, s2 = 5, 7

            in_dim = s1 * k
            out_dim = s2 * k

            dense = Dense(in_dim => out_dim, sigmoid)
            block_dense = BlockDense(in_dim => out_dim, k, sigmoid)

            dense.weight .= false
            for i in 1:k
                dense.weight[(1:s2) .+ s2 * (i-1), (1:s1) .+ s1 * (i-1)] .= block_dense.weight[:,:,i]
            end

            x = rand(Float32, in_dim)
            @test dense(x) ≈ block_dense(x)
        end

    end

end
