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

    @testset "BackendMap resolves per-primitive" begin
        mb = Onion.BackendMap(_ -> DefaultBackend())
        x = randn(Float32, 8, 3)
        W = randn(Float32, 12, 8)
        b = randn(Float32, 12)
        y = Onion.linear(mb, x, W, b)
        @test size(y) == (12, 3)
    end

    @testset "BackendMap with selective dispatch" begin
        mb = Onion.BackendMap(p -> DefaultBackend())
        x = randn(Float32, 8, 3)
        y = Onion.softmax(mb, x)
        @test size(y) == (8, 3)
        @test all(isapprox.(sum(y; dims=1), 1f0; atol=1f-6))
    end

    @testset "backend types" begin
        @test DefaultBackend() isa Onion.Backend
        @test NNkernelsBackend() isa Onion.Backend
        @test TillitBackend() isa Onion.Backend
    end
end
