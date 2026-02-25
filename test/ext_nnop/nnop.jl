using Pkg
Pkg.activate(temp=true)
Pkg.add("CUDA")
Pkg.add("cuTile", version="0.0.4")

@testset "NNop Extension" begin
    @testset "softmax matches DefaultBackend" begin
        x = randn(Float32, 16, 8)
        ref = Onion.softmax(DefaultBackend(), x)
        y = Onion.softmax(NNopBackend(), x)
        @test y ≈ ref
    end

    @testset "rms_norm matches DefaultBackend" begin
        x = randn(Float32, 16, 8)
        w = ones(Float32, 16)
        ref = Onion.rms_norm(DefaultBackend(), x, w; eps=1f-5, offset=0f0)
        y = Onion.rms_norm(NNopBackend(), x, w; eps=1f-5, offset=0f0)
        @test y ≈ ref
    end

    @testset "layer_norm matches DefaultBackend" begin
        x = randn(Float32, 16, 8)
        w = ones(Float32, 16)
        b = zeros(Float32, 16)
        ref = Onion.layer_norm(DefaultBackend(), x, w, b; eps=1f-6)
        y = Onion.layer_norm(NNopBackend(), x, w, b; eps=1f-6)
        @test y ≈ ref
    end

    @testset "attention matches DefaultBackend" begin
        q = randn(Float32, 8, 4, 2, 1)
        k = randn(Float32, 8, 4, 2, 1)
        v = randn(Float32, 8, 4, 2, 1)
        ref = Onion.attention(DefaultBackend(), q, k, v)
        y = Onion.attention(NNopBackend(), q, k, v)
        @test y ≈ ref rtol=1f-3
    end
end
