@testset "backend dispatch" begin

    @testset "backend! sets global backend" begin
        Onion.backend!(DefaultBackend())
        @test Onion.backend() isa DefaultBackend
    end

    @testset "withbackend scoping" begin
        Onion.backend!(DefaultBackend())
        # Outer scope: DefaultBackend
        @test Onion.backend() isa DefaultBackend

        # withbackend temporarily changes backend
        Onion.withbackend(NNkernelsBackend()) do
            @test Onion.backend() isa NNkernelsBackend
        end

        # After scope: back to DefaultBackend
        @test Onion.backend() isa DefaultBackend
    end

    @testset "Rules(backend=...) overrides global" begin
        Onion.backend!(DefaultBackend())
        x = randn(Float32, 8, 3)
        W = randn(Float32, 12, 8)
        b = randn(Float32, 12)

        # Rules with explicit backend
        r = Onion.Rules(backend=DefaultBackend())
        y = Onion.linear(r, x, W, b)
        @test size(y) == (12, 3)
    end

    @testset "function-based backend selector" begin
        selector = p -> DefaultBackend()
        r = Onion.Rules(backend=selector)
        x = randn(Float32, 8, 3)
        y = Onion.softmax(r, x)
        @test size(y) == (8, 3)
        @test all(isapprox.(sum(y; dims=1), 1f0; atol=1f-6))
    end

    @testset "backend types" begin
        @test DefaultBackend() isa Onion.Backend
        @test NNkernelsBackend() isa Onion.Backend
        @test TillitBackend() isa Onion.Backend
    end
end
