@testset "causal_conv1d primitive" begin
    D, K, B = 8, 4, 2
    T = Float32

    x = randn(T, D, B)
    conv_state = randn(T, D, K, B) .* T(0.1)
    weight = randn(T, D, K) .* T(0.02)
    bias = randn(T, D)

    @testset "with bias and SiLU" begin
        state_copy = copy(conv_state)
        y, state_out = Onion.causal_conv1d(DefaultBackend(), x, conv_state, weight, bias; silu=true)

        @test size(y) == (D, B)
        @test state_out === conv_state  # mutated in-place

        # Verify: last column should be x
        @test conv_state[:, K, :] ≈ x
        # Verify: shifted correctly (column i should be old column i+1 for i < K)
        for i in 1:K-1
            @test conv_state[:, i, :] ≈ state_copy[:, i+1, :]
        end

        # Verify convolution result for one (channel, batch)
        d, b = 1, 1
        dot = sum(conv_state[d, :, b] .* weight[d, :])
        dot += bias[d]
        expected = dot / (1 + exp(-dot))  # SiLU = x * sigmoid(x)
        @test y[d, b] ≈ expected
    end

    @testset "without bias, no SiLU" begin
        state2 = randn(T, D, K, B) .* T(0.1)
        y, _ = Onion.causal_conv1d(DefaultBackend(), x, state2, weight, false; silu=false)

        @test size(y) == (D, B)

        # Verify: simple dot product without bias or activation
        d, b = 1, 1
        expected = sum(state2[d, :, b] .* weight[d, :])
        @test y[d, b] ≈ expected
    end
end
