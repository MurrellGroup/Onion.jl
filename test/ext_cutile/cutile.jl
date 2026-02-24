using CUDA
using cuTile

@testset "cuTile Extension" begin
    @testset "softmax matches DefaultBackend" begin
        x = CUDA.randn(Float32, 16, 8)
        ref = Onion.softmax(DefaultBackend(), x)
        y = Onion.softmax(cuTileBackend(), x)
        @test y ≈ ref
    end

    @testset "rms_norm matches DefaultBackend" begin
        x = CUDA.randn(Float32, 16, 8)
        w = CUDA.ones(Float32, 16)
        ref = Onion.rms_norm(DefaultBackend(), x, w; eps=1f-5, offset=0f0)
        y = Onion.rms_norm(cuTileBackend(), x, w; eps=1f-5, offset=0f0)
        @test y ≈ ref
    end

    @testset "layer_norm matches DefaultBackend" begin
        x = CUDA.randn(Float32, 16, 8)
        w = CUDA.ones(Float32, 16)
        b = CUDA.zeros(Float32, 16)
        ref = Onion.layer_norm(DefaultBackend(), x, w, b; eps=1f-6)
        y = Onion.layer_norm(cuTileBackend(), x, w, b; eps=1f-6)
        @test y ≈ ref
    end

    @testset "linear matches DefaultBackend" begin
        x = CUDA.randn(Float32, 8, 4)
        W = CUDA.randn(Float32, 12, 8)
        b = CUDA.randn(Float32, 12)
        ref = Onion.linear(DefaultBackend(), x, W, b)
        y = Onion.linear(cuTileBackend(), x, W, b)
        @test y ≈ ref
    end

    @testset "attention matches DefaultBackend" begin
        q = CUDA.randn(Float32, 8, 4, 2, 1)
        k = CUDA.randn(Float32, 8, 4, 2, 1)
        v = CUDA.randn(Float32, 8, 4, 2, 1)
        ref = Onion.attention(DefaultBackend(), q, k, v)
        y = Onion.attention(cuTileBackend(), q, k, v)
        @test y ≈ ref rtol=1f-3
    end

    @testset "glu_ffn matches DefaultBackend" begin
        x = CUDA.randn(Float32, 8, 4)
        Wg = CUDA.randn(Float32, 16, 8)
        Wu = CUDA.randn(Float32, 16, 8)
        Wd = CUDA.randn(Float32, 8, 16)
        ref = Onion.glu_ffn(DefaultBackend(), x, Wg, Wu, Wd)
        y = Onion.glu_ffn(cuTileBackend(), x, Wg, Wu, Wd)
        @test y ≈ ref rtol=1f-3
    end

    @testset "multihead_ffn matches DefaultBackend" begin
        Q = CUDA.randn(Float32, 8, 2, 4)
        Kg = CUDA.randn(Float32, 12, 8, 2)
        Ku = CUDA.randn(Float32, 12, 8, 2)
        V = CUDA.randn(Float32, 8, 12, 2)
        ref = Onion.multihead_ffn(DefaultBackend(), Q, Kg, Ku, V, Flux.swish)
        y = Onion.multihead_ffn(cuTileBackend(), Q, Kg, Ku, V, Flux.swish)
        @test y ≈ ref rtol=1f-3
    end
end
