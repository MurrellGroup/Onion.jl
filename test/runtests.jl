using Onion
using Test
using Flux

@testset "Onion.jl" begin
    # Test UNet components
    #=@testset "UNet Components" begin
        # Test TimeEmbedding
        @testset "TimeEmbedding" begin
            time_emb = TimeEmbedding(256, 10, 128)
            t = randn(Float32, 16)
            labels = rand(1:10, 16)

            # Test time-only embedding
            h1 = time_emb(t)
            @test size(h1) == (256, 16)

            # Test time+class embedding
            h2 = time_emb(t, labels)
            @test size(h2) == (256, 16)
        end

        # Test ResidualBlock
        @testset "ResidualBlock" begin
            # Without time embedding
            rb1 = ResidualBlock(64)
            x = randn(Float32, 32, 32, 64, 2)
            y1 = rb1(x)
            @test size(y1) == size(x)

            # With time embedding
            rb2 = ResidualBlock(64, time_emb=true, emb_dim=256)
            t = randn(Float32, 256, 2)
            y2 = rb2(x, t)
            @test size(y2) == size(x)
        end

        # Test EncoderBlock
        @testset "EncoderBlock" begin
            eb = EncoderBlock(3, 64, time_emb=true, emb_dim=256)
            x = randn(Float32, 32, 32, 3, 2)
            t = randn(Float32, 256, 2)

            skip, out = eb(x, t)
            @test size(skip) == (32, 32, 64, 2)
            @test size(out) == (16, 16, 64, 2)
        end

        # Test FlexibleUNet
        @testset "FlexibleUNet Forward Pass" begin
            model = FlexibleUNet(
                in_channels=3,
                out_channels=3,
                depth=4,
                base_channels=16,
                channel_multipliers=[1, 2, 4, 8],
                time_embedding=true,
                num_classes=10,
                embedding_dim=32,
                time_emb_dim=64
            )

            x = randn(Float32, 32, 32, 3, 2)
            t = randn(Float32, 2)
            labels = rand(1:10, 2)

            # Test unconditional forward pass
            y1 = model(x)
            @test size(y1) == (32, 32, 3, 2)

            # Test time-conditioned forward pass
            y2 = model(x, t)
            @test size(y2) == (32, 32, 3, 2)

            # Test time+class-conditioned forward pass
            y3 = model(x, t, labels)
            @test size(y3) == (32, 32, 3, 2)
        end

        # Test helper functions
        @testset "UNet Helpers" begin
            # Test reverse_tuple
            t = (1, 2, 3, 4, 5)
            rt = reverse_tuple(t)
            @test rt == (5, 4, 3, 2, 1)

            # Test process_encoders and process_decoders with a mini-model
            enc1 = EncoderBlock(3, 16)
            enc2 = EncoderBlock(16, 32)
            encoders = (enc1, enc2)

            dec1 = DecoderBlock(32, 16)
            dec2 = DecoderBlock(16, 8)
            decoders = (dec1, dec2)

            x = randn(Float32, 32, 32, 3, 2)

            # Test process_encoders
            x_out, skips = process_encoders(x, encoders)
            @test length(skips) == 2
            @test size(x_out) == (8, 8, 32, 2)

            # Test process_decoders
            rev_skips = reverse_tuple(skips)
            y = process_decoders(x_out, decoders, rev_skips)
            @test size(y) == (32, 32, 8, 2)
        end
    end=#

    @testset "DyT" begin
        dyt = DyT(256)
        x = randn(Float32, 256, 2)
        y = dyt(x)
        @test size(y) == size(x)
    end

    @testset "Attention Tests" begin
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

    @testset "RoPE Tests" begin
        @testset "MultidimRoPE Translational Invariance" begin
            dim, n_heads, n_kv_heads, seqlen, batch_size = 64, 8, 4, 16, 2
            for d_coords in [1, 3, 5]
                t = TransformerBlock(dim, n_heads, n_kv_heads)
                x = randn(Float32, dim, seqlen, batch_size)
                pos = randn(Float32, d_coords, seqlen, batch_size)
                rope = MultidimRoPE() 
                out = t(x; rope=x->rope(x, pos))
                delta = repeat(randn(Float32, d_coords, 1, batch_size), 1, seqlen, 1)
                pos2 = pos + delta
                out2 = t(x; rope=x->rope(x, pos2))
                @test size(out) == size(out2) == (dim, seqlen, batch_size)
                @test all(isfinite.(out))
                @test all(isfinite.(out2))
                @test isapprox(out, out2)
            end
        end
        @testset "STRINGRoPE Translational Invariance" begin
            dim, seq_len, batch_size, n_heads = 384, 6, 2, 8
            for d_coords in [1, 3, 5]
                x = randn(Float32, dim, seq_len, batch_size)
                pos = randn(Float32, d_coords, seq_len, batch_size)
                block = STRINGBlock(TransformerBlock(dim, n_heads), d_coords; init_scale=10) # High init_scale to stress test
                out = block(x; positions=pos)
                diff = repeat(randn(Float32, d_coords, 1, batch_size), 1, seq_len, 1)
                pos2 = pos + diff
                out2 = block(x; positions=pos2)
                @test size(out) == size(out2) == (dim, seq_len, batch_size)
                @test all(isfinite.(out))
                @test all(isfinite.(out2))
                @test isapprox(out, out2) 
                cond = randn(Float32, dim, batch_size)
                block = STRINGBlock(AdaTransformerBlock(dim, dim, n_heads), d_coords; init_scale=10)
                out = block(x; positions=pos, cond)
                out2 = block(x; positions=pos2, cond)
                @test size(out) == size(out2) == (dim, seq_len, batch_size) 
                @test all(isfinite.(out))
                @test all(isfinite.(out2))
                @test isapprox(out, out2)
            end
        end
        @testset "STRINGRoPE, MultidimRoPE and RoPE Equivalence" begin
            # with d_coords=1 and init_scale=0 (for STRING) these should all behave the same
            head_dim, seq_len, batch_size = 16, 8, 2
            rope_std = RoPE(head_dim, seq_len)
            rope_string = STRINGRoPE(head_dim, 1, 1, init_scale=0.0f0)
            rope_multi = MultidimRoPE(theta=10000f0)
            x = randn(Float32, head_dim, seq_len, 1, batch_size)
            positions = reshape(Float32.(0:seq_len-1), 1, seq_len, 1) 
            positions = repeat(positions, 1, 1, batch_size)
            out_std = rope_std(x)
            out_string = rope_string(x, positions)
            out_multi = rope_multi(x, positions)
            @test size(out_std) == size(out_string) == size(out_multi)
            @test isapprox(out_std, out_string, rtol=1e-5)
            @test isapprox(out_std, out_multi, rtol=1e-5)
        end
    end
end