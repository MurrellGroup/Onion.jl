@testset "transformers.jl" begin
    @testset "Self-Attention" begin
        dim, seq_len, batch_size, n_heads = 32, 10, 4, 2
        x = randn(Float32, dim, seq_len, batch_size)
        attn = Attention(dim, n_heads)
        output = attn(x)
        @test size(output) == (dim,seq_len,batch_size)

        # The self attention should also be the same as cross attention with itself
        @test isapprox(output, attn(x,x))
    end

    @testset "Cross-Attention" begin
        dim, seq_len, batch_size, n_heads = 32, 10, 4, 2
        q, k = randn(Float32, dim, seq_len, batch_size), randn(Float32, dim, seq_len, batch_size)
        attn = Attention(dim, n_heads)
        output = attn(q, k)
        @test size(output) == (dim,seq_len,batch_size)

        # Key of different length
        k_len = 12
        k = randn(Float32, dim, k_len, batch_size)
        attn = Attention(dim, n_heads)
        output = attn(q, k)
        @test size(output) == (dim,seq_len,batch_size)
    end

    #=
    @testset "Masking" begin
        dim, seq_len, batch_size, n_heads = 32, 10, 4, 2
        x = randn(Float32, dim, seq_len, batch_size)

        int_mask = 0
        batch_mask = zeros(Float32, seq_len, seq_len) 
        seq_mask = zeros(Float32, seq_len, seq_len, batch_size)

        attn = Attention(dim, n_heads)
        no_mask_output = attn(x)
        int_mask_output = attn(x, mask=int_mask)
        batch_mask_output = attn(x, mask=batch_mask)
        seq_mask_output = attn(x, mask=seq_mask)

        # Zero mask should work and give same output regardless of if it's batch-specific, sequence-specific or just an Int
        @test isapprox(no_mask_output, int_mask_output)
        @test isapprox(no_mask_output, batch_mask_output)
        @test isapprox(no_mask_output, seq_mask_output)

        # Using some arbitrary batch_mask should be identical to a seq_mask with the same mask for for every batch
        rand_batch_mask = randn(Float32, seq_len, seq_len)
        rand_seq_mask = repeat(rand_batch_mask, 1, 1, batch_size) 
        batch_mask_output = attn(x, mask=rand_batch_mask)
        seq_mask_output = attn(x, mask=rand_seq_mask)

        @test isapprox(attn(x, mask=rand_batch_mask), attn(x, mask=rand_seq_mask))
    end

    @testset "Masking Invariance" begin
        dim, seq_len, batch_size, n_heads = 32, 10, 4, 2
        x = randn(Float32, dim, seq_len, batch_size)
        # We create a batch_mask with only zeros except for -Inf for the location (2,1)
        # This means that first token should be invariant to what exists at position 2
        x_mod = copy(x)
        x_mod[:, 2, :] = repeat(rand(Float32, dim), 1, batch_size)
        batch_mask = zeros(Float32, seq_len, seq_len)
        batch_mask[2,1] = -Inf32
        attn = Attention(dim, n_heads)
        output = attn(x, mask=batch_mask)
        mod_output = attn(x_mod, mask=batch_mask)
        
        @test isapprox(output[:, 1, :], mod_output[:, 1, :])
        @test !isapprox(output[:, 2, :], mod_output[:, 2, :])

        # Same thing with sequence specific mask 
        dim, seq_len, batch_size, n_heads = 32, 10, 4, 2
        x = randn(Float32, dim, seq_len, batch_size)

        x_mod = copy(x)
        batch_mask = zeros(Float32, seq_len, seq_len, batch_size)

        for batch in 1:batch_size
            x_mod[:, batch+1, batch] = randn(dim)
            batch_mask[batch+1, 1, batch] = -Inf32
        end

        attn = Attention(dim, n_heads)

        output = attn(x, mask=batch_mask)
        mod_output = attn(x_mod, mask=batch_mask)

        @test isapprox(output[:, 1, :], mod_output[:, 1, :])
        @test !isapprox(output, mod_output)

        # Same thing with cross attention and sequence specific mask
        dim, q_seq_len, k_seq_len, batch_size, n_heads = 32, 10, 12, 4, 2
        q = randn(Float32, dim, q_seq_len, batch_size)
        k = randn(Float32, dim, k_seq_len, batch_size)

        k_mod = copy(k)
        # We make some arbitrary changes to the key 
        k_mod[:, 2, 1] = rand(Float32, dim)
        k_mod[:, 3, 2] = rand(Float32, dim)
        k_mod[:, 4, 3] = rand(Float32, dim)
        k_mod[:, 12, 4] = rand(Float32, dim)

        seq_mask = zeros(Float32, k_seq_len, q_seq_len, batch_size)
        # and make some arbitrary positions in the query unable to attend to those changed key positions 
        seq_mask[2,1,1] = -Inf32 
        seq_mask[3,2,2] = -Inf32 
        seq_mask[4,3,3] = -Inf32 
        seq_mask[12,10,4] = -Inf32 
        
        attn = Attention(dim, n_heads)
        output = attn(q, k, mask=seq_mask)
        
        mod_output = attn(q, k_mod, mask=seq_mask)        
        # The masked positions should be invariant to the changes in the key
        @test isapprox(output[:, 1, 1], mod_output[:, 1, 1])
        @test isapprox(output[:, 2, 2], mod_output[:, 2, 2])
        @test isapprox(output[:, 3, 3], mod_output[:, 3, 3])
        @test isapprox(output[:, 10, 4], mod_output[:, 10, 4])
        @test !isapprox(output, mod_output)
    end
    =#
end
