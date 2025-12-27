@testset "Modulator.jl" begin

    @testset "Single output dimension" begin
        Y = rand(Float32, 7, 5)
        X = rand(Float32, 6, 5)

        m = Modulator(6 => 7)
        @test m.W.weight |> size == (7, 6)
        @test m.shape == 7
        @test m(Y, X) |> size == (7, 5)
    end

    @testset "Multiple output dimensions" begin
        Y = rand(Float32, 5, 7, 5)
        X = rand(Float32, 6, 5)

        @testset "Two dimensional output" begin
            m = Modulator(6 => (5, 7))
            @test m.W.weight |> size == (35, 6)
            @test m.shape == (5, 7)
            @test m(Y, X) |> size == (5, 7, 5)
        end

        @testset "Broadcasting along first dimension" begin
            m = Modulator(6 => (1, 7))
            @test m.W.weight |> size == (7, 6)
            @test m.shape == (1, 7)
            @test m(Y, X) |> size == (5, 7, 5)
        end

        @testset "Broadcasting along second dimension" begin
            m = Modulator(6 => (5, 1))
            @test m.W.weight |> size == (5, 6)
            @test m.shape == (5, 1)
            @test m(Y, X) |> size == (5, 7, 5)
        end
    end

end
