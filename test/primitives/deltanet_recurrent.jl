@testset "deltanet_recurrent_decode primitive" begin
    Dk, Dv, H, B = 4, 4, 2, 1
    T = Float32

    q = randn(T, Dk, H, B)
    k = randn(T, Dk, H, B)
    v = randn(T, Dv, H, B)
    beta = rand(T, H, B)       # in [0, 1]
    gate = randn(T, H, B) .* T(0.1)  # small negative values typical
    state = randn(T, Dk, Dv, H, B) .* T(0.1)

    state_copy = copy(state)

    output, state_out = Onion.deltanet_recurrent_decode(DefaultBackend(), q, k, v, beta, gate, state)

    @testset "output shape" begin
        @test size(output) == (Dv, H, B)
    end

    @testset "state is mutated in-place" begin
        @test state === state_out
        @test state != state_copy  # state should have changed
    end

    @testset "manual verification" begin
        # Verify against manual computation for one head
        h, b = 1, 1
        q_h = q[:, h, b]
        k_h = k[:, h, b]
        v_h = v[:, h, b]
        beta_h = beta[h, b]
        gate_h = gate[h, b]
        S = state_copy[:, :, h, b]

        # Step 1: Decay
        decay = exp(gate_h)
        S_decayed = S .* decay

        # Step 2: S^T @ k
        stk = S_decayed' * k_h

        # Step 3: Delta rule
        delta = beta_h .* (v_h .- stk)

        # Step 4: Rank-1 update
        S_updated = S_decayed .+ k_h * delta'

        # Step 5: Output query
        o_expected = S_updated' * q_h

        @test output[:, h, b] ≈ o_expected
        @test state[:, :, h, b] ≈ S_updated
    end

    @testset "BackendMap dispatch" begin
        s1 = copy(state_copy)
        s2 = copy(state_copy)
        mb = Onion.BackendMap(_ -> DefaultBackend())
        o1 = Onion.deltanet_recurrent_decode(DefaultBackend(), q, k, v, beta, gate, s1)[1]
        o2 = Onion.deltanet_recurrent_decode(mb, q, k, v, beta, gate, s2)[1]
        @test o1 ≈ o2
    end
end
