using OnionProt
using Onion
using Test
using LinearAlgebra
using Statistics

@testset "OnionProt" begin

# ── Pairwise layers ──────────────────────────────────────────────────────────

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
    dim = 16
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
    L, B = 4, 1

    m = TriangularSelfAttentionBlock(seq_dim, pair_dim, seq_hw, pair_hw)
    s = randn(Float32, seq_dim, L, B)
    z = randn(Float32, pair_dim, L, L, B)
    s2, z2 = m(s, z)

    @test size(s2) == (seq_dim, L, B)
    @test size(z2) == (pair_dim, L, L, B)
    @test all(isfinite, s2)
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

# ── Structural layers ────────────────────────────────────────────────────────

@testset "Rigid identity" begin
    r = rigid_identity((5, 2), zeros(Float32, 1); fmt=:quat)
    @test r.rots isa QuatRotation
    @test size(r.rots.quats) == (4, 5, 2)
    @test size(r.trans) == (3, 5, 2)
    @test r.rots.quats[1, 1, 1] ≈ 1f0
    @test all(r.trans .== 0)
end

@testset "apply_rigid round-trip" begin
    q = randn(Float32, 4, 5, 2)
    rot = rot_from_quat(q)
    trans = randn(Float32, 3, 5, 2)
    r = Rigid(rot, trans)

    pts = randn(Float32, 3, 5, 2)
    transformed = apply_rigid(r, pts)
    recovered = invert_apply_rigid(r, transformed)
    @test recovered ≈ pts atol=1e-4
end

@testset "compose" begin
    q1 = randn(Float32, 4, 3, 2)
    q2 = randn(Float32, 4, 3, 2)
    r1 = Rigid(rot_from_quat(q1), randn(Float32, 3, 3, 2))
    r2 = Rigid(rot_from_quat(q2), randn(Float32, 3, 3, 2))

    r12 = compose(r1, r2)
    pts = randn(Float32, 3, 3, 2)
    via_compose = apply_rigid(r12, pts)
    via_chain = apply_rigid(r1, apply_rigid(r2, pts))
    @test via_compose ≈ via_chain atol=1e-4
end

@testset "PointProjection" begin
    c_hidden, num_points, no_heads = 32, 4, 2
    L, B = 5, 2

    m = PointProjection(c_hidden, num_points, no_heads)
    s = randn(Float32, c_hidden, L, B)
    r = rigid_identity((L, B), s; fmt=:quat)

    pts = m(s, r)
    @test size(pts) == (3, num_points, no_heads, L, B)
    @test all(isfinite, pts)
end

@testset "ESMFoldIPA" begin
    c_s, c_z, c_hidden = 32, 16, 8
    no_heads, no_qk_points, no_v_points = 2, 2, 2
    L, B = 4, 1

    m = ESMFoldIPA(c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points)
    s = randn(Float32, c_s, L, B)
    z = randn(Float32, c_z, L, L, B)
    r = rigid_identity((L, B), s; fmt=:quat)
    mask = ones(Float32, L, B)

    y = m(s, z, r, mask)
    @test size(y) == (c_s, L, B)
    @test all(isfinite, y)
end

@testset "MultimerInvariantPointAttention" begin
    c_s, c_z, c_hidden = 32, 16, 8
    no_heads, no_qk_points, no_v_points = 2, 2, 2
    L, B = 4, 1

    m = MultimerInvariantPointAttention(c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points)
    s = randn(Float32, c_s, L, B)
    z = randn(Float32, c_z, L, L, B)
    r = rigid_identity((L, B), s; fmt=:quat)
    mask = ones(Float32, L, B)

    y = m(s, z, r, mask)
    @test size(y) == (c_s, L, B)
    @test all(isfinite, y)
end

@testset "BackboneUpdate" begin
    c_s = 32
    m = BackboneUpdate(c_s)
    s = randn(Float32, c_s, 5, 2)
    y = m(s)
    @test size(y) == (6, 5, 2)
end

@testset "AngleResnet" begin
    c_in, c_hidden, no_blocks, no_angles = 32, 16, 2, 7
    eps = 1f-8
    L, B = 5, 2

    m = AngleResnet(c_in, c_hidden, no_blocks, no_angles, eps)
    s = randn(Float32, c_in, L, B)
    s_initial = randn(Float32, c_in, L, B)

    unnormalized, normalized = m(s, s_initial)
    @test size(unnormalized) == (2, no_angles, L, B)
    @test size(normalized) == (2, no_angles, L, B)
    norms = sqrt.(sum(normalized .^ 2; dims=1))
    @test all(x -> isapprox(x, 1f0; atol=1e-4), norms)
end

@testset "RelativePosition" begin
    bins, c_z = 16, 12
    m = RelativePosition(bins, c_z)
    L, B = 5, 2
    residx = repeat(collect(1:L), 1, B)

    emb = m(residx)
    @test size(emb) == (c_z, L, L, B)
    @test all(isfinite, emb)
end

@testset "distogram" begin
    L, B = 5, 2
    coords = randn(Float32, 3, 3, L, B)
    bins = distogram(coords, 3.375f0, 21.375f0, 15)
    @test size(bins) == (L, L, B)
    @test eltype(bins) <: Integer
    @test all(b -> 0 ≤ b ≤ 14, bins)
end

@testset "Framemover" begin
    dim = 32
    L, B = 5, 2
    fm = Framemover(dim)
    frames = rigid_identity((L, B), zeros(Float32, 1); fmt=:quat)
    x = randn(Float32, dim, L, B)

    new_frames = fm(frames, x)
    @test new_frames isa Rigid
    @test size(new_frames.trans) == (3, L, B)
    @test all(isfinite, new_frames.trans)
end

end # @testset "OnionProt"
