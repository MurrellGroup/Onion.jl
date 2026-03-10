using NNlib: swish, relu

@testset "glu_ffn primitive" begin
    d_in, d_hidden, batch = 8, 16, 3
    x = randn(Float32, d_in, batch)
    W_gate = randn(Float32, d_hidden, d_in)
    W_up   = randn(Float32, d_hidden, d_in)
    W_down = randn(Float32, d_in, d_hidden)

    @testset "output shape" begin
        y = Onion.glu_ffn(DefaultBackend(), x, W_gate, W_up, W_down)
        @test size(y) == (d_in, batch)
    end

    @testset "correctness (manual reference)" begin
        y = Onion.glu_ffn(DefaultBackend(), x, W_gate, W_up, W_down, swish)
        ref = W_down * (swish.(W_gate * x) .* (W_up * x))
        @test y ≈ ref
    end

    @testset "custom activation" begin
        y = Onion.glu_ffn(DefaultBackend(), x, W_gate, W_up, W_down, relu)
        ref = W_down * (relu.(W_gate * x) .* (W_up * x))
        @test y ≈ ref
    end

    @testset "top-level dispatch" begin
        y_explicit = Onion.glu_ffn(DefaultBackend(), x, W_gate, W_up, W_down)
        y_implicit = Onion.glu_ffn(x, W_gate, W_up, W_down)
        @test y_explicit ≈ y_implicit
    end
end
