@testset "Connection layers" begin
    @testset "SkipConnection" begin
        sc = Onion.SkipConnection(+)
        layer = Linear(8 => 8)
        x = randn(Float32, 8, 5)
        y = sc(layer, x)
        @test size(y) == (8, 5)
        @test y ≈ layer(x) + x
    end

    @testset "ResidualConnection" begin
        rc = ResidualConnection()
        layer = Linear(8 => 8)
        x = randn(Float32, 8, 5)
        y = rc(layer, x)
        @test size(y) == (8, 5)
        @test y ≈ layer(x) + x
    end

    @testset "GHC construction" begin
        ghc = GHC(3, 2)
        @test size(ghc.down) == (3, 2)
        @test size(ghc.side) == (3, 3)
        @test size(ghc.up) == (2, 3)
    end

    @testset "GHC n >= m assertion" begin
        @test_throws ArgumentError GHC(1, 2)
    end

    @testset "GHC forward" begin
        ghc = GHC(3, 2)
        layer = Linear(8 => 8)
        # Hidden state: 12 = 4 * 3 (d=4, n=3)
        h = randn(Float32, 12, 5)
        y = ghc(layer, h)
        @test size(y) == (12, 5)
    end

    @testset "GHC curried form" begin
        ghc = GHC(3, 2)
        layer = Linear(8 => 8)
        h = randn(Float32, 12, 5)
        curried = ghc(layer)
        y = curried(h)
        @test size(y) == (12, 5)
    end

    @testset "VirtualWidthNetwork" begin
        layer = Linear(8 => 8)
        model = VirtualWidthNetwork(layer, 3, 2)
        x = randn(Float32, 12, 5)
        y = model(x)
        @test size(y) == (12, 5)
    end
end
