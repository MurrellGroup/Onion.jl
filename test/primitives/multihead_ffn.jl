using Einops: einsum, @einops_str
using NNlib: swish

@testset "multihead_ffn primitive" begin
    d_head, n_heads, d_inter, batch = 8, 2, 12, 3

    Q = randn(Float32, d_head, n_heads, batch)
    K_gate = randn(Float32, d_inter, d_head, n_heads)
    K_up   = randn(Float32, d_inter, d_head, n_heads)
    V      = randn(Float32, d_head, d_inter, n_heads)

    @testset "output shape" begin
        y = Onion.multihead_ffn(DefaultBackend(), Q, K_gate, K_up, V, swish)
        @test size(y) == (d_head, n_heads, batch)
    end

    @testset "correctness (manual reference)" begin
        act = swish
        y = Onion.multihead_ffn(DefaultBackend(), Q, K_gate, K_up, V, act)
        ϕ = act.(einsum(K_gate, Q, einops"i dh h, dh h ... -> i h ...")) .*
                 einsum(K_up, Q, einops"i dh h, dh h ... -> i h ...")
        ref = einsum(V, ϕ, einops"dh i h, i h ... -> dh h ...")
        @test y ≈ ref
    end

    @testset "top-level dispatch" begin
        y_explicit = Onion.multihead_ffn(DefaultBackend(), Q, K_gate, K_up, V, swish)
        y_implicit = Onion.multihead_ffn(Q, K_gate, K_up, V, swish)
        @test y_explicit ≈ y_implicit
    end
end
