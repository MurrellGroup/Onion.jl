@testset "embedding.jl" begin
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