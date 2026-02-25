@testset "training_mode toggle" begin
    prev = training_mode()
    training_mode!(true)
    @test training_mode() == true
    training_mode!(false)
    @test training_mode() == false
    training_mode!(prev)
end

@testset "SharedDropout" begin
    @testset "identity when not training" begin
        training_mode!(false)
        drop = SharedDropout(0.5, [2])
        x = randn(Float32, 8, 4, 2)
        @test drop(x) == x
    end

    @testset "identity when rate=0" begin
        training_mode!(true)
        drop = SharedDropout(0.0, [2])
        x = randn(Float32, 8, 4, 2)
        @test drop(x) == x
        training_mode!(false)
    end

    @testset "shape preserved in training" begin
        training_mode!(true)
        drop = SharedDropout(0.1, [2])
        x = randn(Float32, 8, 4, 2)
        y = drop(x)
        @test size(y) == size(x)
        training_mode!(false)
    end
end

@testset "Transition" begin
    @testset "basic forward" begin
        dim = 16
        t = Transition(dim)
        x = randn(Float32, dim, 5, 2)
        y = t(x)
        @test size(y) == (dim, 5, 2)
        @test all(isfinite, y)
    end

    @testset "custom hidden and out_dim" begin
        t = Transition(16, 32; out_dim=8)
        x = randn(Float32, 16, 5, 2)
        y = t(x)
        @test size(y) == (8, 5, 2)
    end
end

@testset "SequenceToPair" begin
    c_s, c_inner, c_z = 16, 8, 12
    m = SequenceToPair(c_s, c_inner, c_z)
    s = randn(Float32, c_s, 5, 2)
    z = m(s)
    @test size(z) == (c_z, 5, 5, 2)
    @test all(isfinite, z)
end

@testset "PairToSequence" begin
    c_z, num_heads = 12, 4
    m = PairToSequence(c_z, num_heads)
    z = randn(Float32, c_z, 5, 5, 2)
    h = m(z)
    @test size(h) == (num_heads, 5, 5, 2)
    @test all(isfinite, h)
end

@testset "ResidueMLP" begin
    dim, inner = 16, 32
    m = ResidueMLP(dim, inner)
    x = randn(Float32, dim, 5, 5, 2)
    y = m(x)
    @test size(y) == (dim, 5, 5, 2)
    @test all(isfinite, y)
end

@testset "OuterProductMean" begin
    c_in, c_hidden, c_out = 8, 4, 12
    m = OuterProductMean(c_in, c_hidden, c_out)
    S, N, B = 3, 5, 2
    x = randn(Float32, c_in, S, N, B)
    mask = ones(Float32, S, N, B)
    y = m(x, mask)
    @test size(y) == (c_out, N, N, B)
    @test all(isfinite, y)
end

@testset "PairWeightedAveraging" begin
    c_m, c_z, c_h, num_heads = 16, 12, 4, 2
    m = PairWeightedAveraging(c_m, c_z, c_h, num_heads)
    S, N, B = 3, 5, 2
    x = randn(Float32, c_m, S, N, B)
    z = randn(Float32, c_z, N, N, B)
    mask = ones(Float32, N, N, B)
    y = m(x, z, mask)
    @test size(y) == (c_m, S, N, B)
    @test all(isfinite, y)
end

@testset "TriangleMultiplicativeUpdate" begin
    c_z, c_hidden = 16, 8
    L, B = 5, 2

    @testset "outgoing" begin
        m = TriangleMultiplicationOutgoing(c_z, c_hidden)
        z = randn(Float32, c_z, L, L, B)
        y = m(z)
        @test size(y) == (c_z, L, L, B)
        @test all(isfinite, y)
    end

    @testset "incoming" begin
        m = TriangleMultiplicationIncoming(c_z, c_hidden)
        z = randn(Float32, c_z, L, L, B)
        y = m(z)
        @test size(y) == (c_z, L, L, B)
        @test all(isfinite, y)
    end

    @testset "with mask" begin
        m = TriangleMultiplicationOutgoing(c_z, c_hidden)
        z = randn(Float32, c_z, L, L, B)
        mask = ones(Float32, L, L, B)
        y = m(z; mask)
        @test size(y) == (c_z, L, L, B)
    end
end

@testset "BGTriangleMultiplication" begin
    dim = 16
    L, B = 5, 2
    mask = ones(Float32, L, L, B)

    @testset "outgoing" begin
        m = BGTriangleMultiplicationOutgoing(dim)
        z = randn(Float32, dim, L, L, B)
        y = m(z; mask)
        @test size(y) == (dim, L, L, B)
        @test all(isfinite, y)
    end

    @testset "incoming" begin
        m = BGTriangleMultiplicationIncoming(dim)
        z = randn(Float32, dim, L, L, B)
        y = m(z; mask)
        @test size(y) == (dim, L, L, B)
        @test all(isfinite, y)
    end
end

@testset "MiniTriangularUpdate" begin
    dim = 16  # must be divisible by 4
    L, B = 5, 2
    m = MiniTriangularUpdate(dim)
    z = randn(Float32, dim, L, L, B)
    mask = ones(Float32, L, L, B)
    y = m(z; mask)
    @test size(y) == (dim, L, L, B)
    @test all(isfinite, y)
end

@testset "TriangleAttention" begin
    c_in, c_hidden, no_heads = 16, 4, 4
    L, B = 5, 2

    @testset "starting node" begin
        m = TriangleAttentionStartingNode(c_in, c_hidden, no_heads)
        x = randn(Float32, c_in, L, L, B)
        y = m(x)
        @test size(y) == (c_in, L, L, B)
        @test all(isfinite, y)
    end

    @testset "ending node" begin
        m = TriangleAttentionEndingNode(c_in, c_hidden, no_heads)
        x = randn(Float32, c_in, L, L, B)
        y = m(x)
        @test size(y) == (c_in, L, L, B)
        @test all(isfinite, y)
    end
end

@testset "AttentionPairBias" begin
    c_s, c_z, num_heads = 32, 16, 4
    L, B = 5, 2

    @testset "with pair bias computation" begin
        m = AttentionPairBias(c_s, c_z, num_heads)
        s = randn(Float32, c_s, L, B)
        z = randn(Float32, c_z, L, L, B)
        mask = ones(Float32, L, B)
        y = m(s, z, mask)
        @test size(y) == (c_s, L, B)
        @test all(isfinite, y)
    end

    @testset "without pair bias computation" begin
        m = AttentionPairBias(c_s, c_z, num_heads; compute_pair_bias=false)
        s = randn(Float32, c_s, L, B)
        z = randn(Float32, num_heads, L, L, B)
        mask = ones(Float32, L, B)
        y = m(s, z, mask)
        @test size(y) == (c_s, L, B)
    end
end

@testset "ESMFoldAttention" begin
    embed_dim, num_heads = 32, 4
    head_width = embed_dim ÷ num_heads
    L, B = 5, 2

    @testset "basic forward" begin
        m = ESMFoldAttention(embed_dim, num_heads, head_width)
        x = randn(Float32, embed_dim, L, B)
        y, extra = m(x)
        @test size(y) == (embed_dim, L, B)
        @test extra === nothing
        @test all(isfinite, y)
    end

    @testset "with bias" begin
        m = ESMFoldAttention(embed_dim, num_heads, head_width)
        x = randn(Float32, embed_dim, L, B)
        bias = randn(Float32, num_heads, L, L, B)
        y, _ = m(x; bias)
        @test size(y) == (embed_dim, L, B)
    end

    @testset "gated" begin
        m = ESMFoldAttention(embed_dim, num_heads, head_width; gated=true)
        x = randn(Float32, embed_dim, L, B)
        y, _ = m(x)
        @test size(y) == (embed_dim, L, B)
    end
end

@testset "TriangularSelfAttentionBlock" begin
    seq_dim, pair_dim = 32, 16
    seq_hw, pair_hw = 8, 4
    L, B = 4, 1  # small for speed

    m = TriangularSelfAttentionBlock(seq_dim, pair_dim, seq_hw, pair_hw)
    s = randn(Float32, seq_dim, L, B)
    z = randn(Float32, pair_dim, L, L, B)
    s2, z2 = m(s, z)

    @test size(s2) == (seq_dim, L, B)
    @test size(z2) == (pair_dim, L, L, B)
    @test all(isfinite, s2)
    @test all(isfinite, z2)
end

@testset "PairformerLayer" begin
    token_s, token_z = 32, 16
    L, B = 4, 1  # small for speed

    m = PairformerLayer(token_s, token_z; num_heads=4, pairwise_head_width=4, pairwise_num_heads=4)
    s = randn(Float32, token_s, L, B)
    z = randn(Float32, token_z, L, L, B)
    mask = ones(Float32, L, B)
    pair_mask = ones(Float32, L, L, B)

    s2, z2 = m(s, z, mask, pair_mask)
    @test size(s2) == (token_s, L, B)
    @test size(z2) == (token_z, L, L, B)
    @test all(isfinite, s2)
    @test all(isfinite, z2)
end

@testset "PairformerNoSeqLayer" begin
    token_z = 16
    L, B = 4, 1

    m = PairformerNoSeqLayer(token_z; pairwise_head_width=4, pairwise_num_heads=4)
    z = randn(Float32, token_z, L, L, B)
    pair_mask = ones(Float32, L, L, B)

    z2 = m(z, pair_mask)
    @test size(z2) == (token_z, L, L, B)
    @test all(isfinite, z2)
end

@testset "MiniformerLayer" begin
    token_s, token_z = 32, 16
    L, B = 4, 1

    m = MiniformerLayer(token_s, token_z; num_heads=4)
    s = randn(Float32, token_s, L, B)
    z = randn(Float32, token_z, L, L, B)
    mask = ones(Float32, L, B)
    pair_mask = ones(Float32, L, L, B)

    s2, z2 = m(s, z, mask, pair_mask)
    @test size(s2) == (token_s, L, B)
    @test size(z2) == (token_z, L, L, B)
    @test all(isfinite, s2)
    @test all(isfinite, z2)
end

@testset "MiniformerNoSeqLayer" begin
    token_z = 16
    L, B = 4, 1

    m = MiniformerNoSeqLayer(token_z)
    z = randn(Float32, token_z, L, L, B)
    pair_mask = ones(Float32, L, L, B)

    z2 = m(z, pair_mask)
    @test size(z2) == (token_z, L, L, B)
    @test all(isfinite, z2)
end

@testset "PairformerModule" begin
    token_s, token_z = 32, 16
    L, B = 4, 1

    m = PairformerModule(token_s, token_z, 2; num_heads=4, pairwise_head_width=4, pairwise_num_heads=4)
    s = randn(Float32, token_s, L, B)
    z = randn(Float32, token_z, L, L, B)
    mask = ones(Float32, L, B)
    pair_mask = ones(Float32, L, L, B)

    s2, z2 = m(s, z, mask, pair_mask)
    @test size(s2) == (token_s, L, B)
    @test size(z2) == (token_z, L, L, B)
    @test length(m.layers) == 2
end

@testset "MiniformerModule" begin
    token_s, token_z = 32, 16
    L, B = 4, 1

    m = MiniformerModule(token_s, token_z, 2; num_heads=4)
    s = randn(Float32, token_s, L, B)
    z = randn(Float32, token_z, L, L, B)
    mask = ones(Float32, L, B)
    pair_mask = ones(Float32, L, L, B)

    s2, z2 = m(s, z, mask, pair_mask)
    @test size(s2) == (token_s, L, B)
    @test size(z2) == (token_z, L, L, B)
    @test length(m.layers) == 2
end
