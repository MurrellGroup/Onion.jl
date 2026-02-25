@testset "Rigid identity" begin
    @testset "quaternion format" begin
        r = rigid_identity((5, 2), zeros(Float32, 1); fmt=:quat)
        @test r.rots isa QuatRotation
        @test size(r.rots.quats) == (4, 5, 2)
        @test size(r.trans) == (3, 5, 2)
        # Identity quaternion: [1, 0, 0, 0]
        @test r.rots.quats[1, 1, 1] ≈ 1f0
        @test r.rots.quats[2, 1, 1] ≈ 0f0
        @test all(r.trans .== 0)
    end

    @testset "rotation matrix format" begin
        r = rigid_identity((5, 2), zeros(Float32, 1); fmt=:rot_mat)
        @test r.rots isa RotMatRotation
        @test size(get_rot_mats(r.rots)) == (3, 3, 5, 2)
        @test size(r.trans) == (3, 5, 2)
    end
end

@testset "apply_rigid" begin
    @testset "identity preserves points" begin
        r = rigid_identity((5, 2), zeros(Float32, 1); fmt=:quat)
        pts = randn(Float32, 3, 5, 2)
        result = apply_rigid(r, pts)
        @test size(result) == (3, 5, 2)
        @test result ≈ pts atol=1e-5
    end

    @testset "translation only" begin
        rot = rotation_identity((2,), zeros(Float32, 1); fmt=:rot_mat)
        trans = Float32[1.0; 2.0; 3.0;;]  # (3, 1) → broadcast to (3, 2)
        trans = repeat(trans, 1, 2)
        r = Rigid(rot, trans)
        pts = zeros(Float32, 3, 2)
        result = apply_rigid(r, pts)
        @test result[1, 1] ≈ 1f0
        @test result[2, 1] ≈ 2f0
        @test result[3, 1] ≈ 3f0
    end

    @testset "extra point dims broadcast correctly" begin
        r = rigid_identity((5, 2), zeros(Float32, 1); fmt=:quat)
        # Points with extra dims between spatial and batch: (3, P, H, L, B)
        pts = randn(Float32, 3, 4, 3, 5, 2)
        result = apply_rigid(r, pts)
        @test size(result) == (3, 4, 3, 5, 2)
        @test result ≈ pts atol=1e-5
    end
end

@testset "invert_apply_rigid round-trip" begin
    # Random rotation via quaternion
    q = randn(Float32, 4, 5, 2)
    rot = rot_from_quat(q)
    trans = randn(Float32, 3, 5, 2)
    r = Rigid(rot, trans)

    pts = randn(Float32, 3, 5, 2)
    transformed = apply_rigid(r, pts)
    recovered = invert_apply_rigid(r, transformed)
    @test recovered ≈ pts atol=1e-4

    @testset "with extra dims" begin
        pts_5d = randn(Float32, 3, 4, 3, 5, 2)
        transformed = apply_rigid(r, pts_5d)
        recovered = invert_apply_rigid(r, transformed)
        @test recovered ≈ pts_5d atol=1e-4
    end
end

@testset "compose" begin
    q1 = randn(Float32, 4, 3, 2)
    q2 = randn(Float32, 4, 3, 2)
    r1 = Rigid(rot_from_quat(q1), randn(Float32, 3, 3, 2))
    r2 = Rigid(rot_from_quat(q2), randn(Float32, 3, 3, 2))

    r12 = compose(r1, r2)
    @test r12.rots isa RotMatRotation
    @test size(get_rot_mats(r12.rots)) == (3, 3, 3, 2)
    @test size(r12.trans) == (3, 3, 2)

    # Verify: apply(compose(r1,r2), p) ≈ apply(r1, apply(r2, p))
    pts = randn(Float32, 3, 3, 2)
    via_compose = apply_rigid(r12, pts)
    via_chain = apply_rigid(r1, apply_rigid(r2, pts))
    @test via_compose ≈ via_chain atol=1e-4
end

@testset "compose_q_update_vec" begin
    r = rigid_identity((5, 2), zeros(Float32, 1); fmt=:quat)
    update = randn(Float32, 6, 5, 2) .* 0.01f0
    r2 = compose_q_update_vec(r, update)
    @test r2.rots isa QuatRotation
    @test size(r2.rots.quats) == (4, 5, 2)
    @test size(r2.trans) == (3, 5, 2)
    # Small update from identity should remain close to identity
    @test r2.rots.quats[1, 1, 1] > 0.9f0  # real part still dominant
end

@testset "scale_translation" begin
    q = randn(Float32, 4, 3)
    r = Rigid(rot_from_quat(q), randn(Float32, 3, 3))
    r2 = scale_translation(r, 10f0)
    @test r2.trans ≈ r.trans .* 10f0
    @test get_rot_mats(r2.rots) ≈ get_rot_mats(r.rots)
end

@testset "to_tensor_7" begin
    r = rigid_identity((5, 2), zeros(Float32, 1); fmt=:quat)
    t7 = to_tensor_7(r)
    @test size(t7) == (7, 5, 2)
    # First 4 = quaternion, last 3 = translation
    @test t7[1, 1, 1] ≈ 1f0  # quat real part
    @test t7[5:7, 1, 1] ≈ zeros(Float32, 3) atol=1e-6
end

@testset "to_tensor_4x4" begin
    r = rigid_identity((5, 2), zeros(Float32, 1); fmt=:quat)
    t44 = to_tensor_4x4(r)
    @test size(t44) == (4, 4, 5, 2)
    @test t44[1, 1, 1, 1] ≈ 1f0
    @test t44[4, 4, 1, 1] ≈ 1f0
    @test t44[4, 1, 1, 1] ≈ 0f0
end

@testset "rigid_index" begin
    r = rigid_identity((5, 2), zeros(Float32, 1); fmt=:quat)
    r_sub = rigid_index(r, 1:3, 1)
    @test size(r_sub.trans) == (3, 3)
end

@testset "quat_to_rot_first" begin
    # Identity quaternion → identity matrix
    q = Float32[1; 0; 0; 0;;]
    rot = quat_to_rot_first(q)
    @test size(rot) == (3, 3, 1)
    @test rot[:, :, 1] ≈ Float32[1 0 0; 0 1 0; 0 0 1] atol=1e-6
end

@testset "rot_matmul_first" begin
    I3 = Float32[1 0 0; 0 1 0; 0 0 1]
    a = reshape(I3, 3, 3, 1)
    b = randn(Float32, 3, 3, 1)
    c = rot_matmul_first(a, b)
    @test c ≈ b atol=1e-6
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

@testset "PointProjectionMultimer" begin
    c_hidden, num_points, no_heads = 32, 4, 2
    L, B = 5, 2

    m = PointProjectionMultimer(c_hidden, num_points, no_heads)
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
    @test all(isfinite, unnormalized)
    @test all(isfinite, normalized)

    # Check normalization: each angle vector should have unit norm along dim 1
    norms = sqrt.(sum(normalized .^ 2; dims=1))
    @test all(x -> isapprox(x, 1f0; atol=1e-4), norms)
end

@testset "StructureModuleTransition" begin
    c, num_layers = 32, 2
    m = Onion.StructureModuleTransition(c, num_layers, 0f0)
    s = randn(Float32, c, 5, 2)
    y = m(s)
    @test size(y) == (c, 5, 2)
    @test all(isfinite, y)
end

@testset "RelativePosition" begin
    bins, c_z = 16, 12
    m = RelativePosition(bins, c_z)
    L, B = 5, 2
    residx = repeat(collect(1:L), 1, B)  # (L, B)

    emb = m(residx)
    @test size(emb) == (c_z, L, L, B)
    @test all(isfinite, emb)
end

@testset "distogram" begin
    L, B = 5, 2
    # coords: (3, 3_atoms, L, B) — N, Ca, C positions
    coords = randn(Float32, 3, 3, L, B)
    bins = distogram(coords, 3.375f0, 21.375f0, 15)
    @test size(bins) == (L, L, B)
    @test eltype(bins) <: Integer
    @test all(b -> 0 ≤ b ≤ 14, bins)
end

@testset "cross_first" begin
    a = randn(Float32, 3, 5)
    b = randn(Float32, 3, 5)
    c = Onion.cross_first(a, b)
    @test size(c) == (3, 5)
    # Cross product is perpendicular to both inputs
    dots_a = sum(a .* c; dims=1)
    dots_b = sum(b .* c; dims=1)
    @test all(x -> abs(x) < 1e-4, dots_a)
    @test all(x -> abs(x) < 1e-4, dots_b)
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
