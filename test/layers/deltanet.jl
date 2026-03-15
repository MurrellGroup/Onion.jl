@testset "DeltaNet layer" begin
    @testset "construction" begin
        layer = DeltaNet(hidden_size=64, n_k_heads=2, n_v_heads=4, head_dim=8)
        @test layer.head_dim == 8
        @test layer.n_k_heads == 2
        @test layer.n_v_heads == 4
        @test layer.kernel_size == 4
    end

    @testset "cache creation" begin
        layer = DeltaNet(hidden_size=64, n_k_heads=2, n_v_heads=4, head_dim=8)
        cache = deltanet_cache(layer, 2)
        d_conv = 2 * 2 * 8 + 4 * 8  # 2*k_dim + v_dim
        @test size(cache.conv_state) == (d_conv, 4, 2)
        @test size(cache.recurrent_state) == (8, 8, 4, 2)
    end

    @testset "forward pass (single token)" begin
        layer = DeltaNet(hidden_size=64, n_k_heads=2, n_v_heads=4, head_dim=8)
        cache = deltanet_cache(layer, 1)
        x = randn(Float32, 64, 1)
        y = Onion.decode(layer, x; cache)
        @test size(y) == (64, 1)
    end

    @testset "forward pass (batch)" begin
        layer = DeltaNet(hidden_size=64, n_k_heads=2, n_v_heads=4, head_dim=8)
        cache = deltanet_cache(layer, 3)
        x = randn(Float32, 64, 3)
        y = Onion.decode(layer, x; cache)
        @test size(y) == (64, 3)
    end

    @testset "multiple decode steps update state" begin
        layer = DeltaNet(hidden_size=64, n_k_heads=2, n_v_heads=4, head_dim=8)
        cache = deltanet_cache(layer, 1)
        s0 = copy(cache.recurrent_state)
        layer(randn(Float32, 64, 1); cache)
        @test cache.recurrent_state != s0  # state should change
        s1 = copy(cache.recurrent_state)
        layer(randn(Float32, 64, 1); cache)
        @test cache.recurrent_state != s1  # state should change again
    end
end
