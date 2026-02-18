using OnionStyle
using Test
using LinearAlgebra

@testset "OnionStyle" begin

    @testset "postfix operators" begin
        @testset "inverse ⁻¹" begin
            @test (2)⁻¹ == 0.5
            @test (4.0)⁻¹ == 0.25
            M = [1.0 2.0; 3.0 4.0]
            @test (M)⁻¹ ≈ inv(M)
        end

        @testset "transpose ᵀ" begin
            M = [1 2; 3 4]
            @test (M)ᵀ == transpose(M)
            @test (M)ᵀ == [1 3; 2 4]
        end

        @testset "transpose ᵀ higher dims" begin
            A = rand(2, 3, 4)
            B = (A)ᵀ
            @test size(B) == (3, 2, 4)
            @test B[2, 1, 3] == A[1, 2, 3]
        end

        @testset "hermitian ᴴ" begin
            M = [1+im 2+im; 3+im 4+im]
            @test (M)ᴴ == LinearAlgebra.hermitian(M)
        end

        @testset "FixPostfix currying" begin
            A = [1.0 2.0; 3.0 4.0]
            B = [5.0 6.0; 7.0 8.0]
            @test (A)ᵀ(B) == transpose(A) * B
        end
    end

    @testset "type conversion →" begin
        @testset "number conversion" begin
            @test (1 → Float32) isa Float32
            @test (1 → Float64) == 1.0
        end

        @testset "array conversion" begin
            x = [1, 2, 3]
            y = x → Float32
            @test eltype(y) == Float32
            @test y == Float32[1, 2, 3]
        end

        @testset "partial application" begin
            to_f32 = →(Float32)
            @test to_f32(1) isa Float32
            @test to_f32([1, 2]) |> eltype == Float32
        end
    end

    @testset "Optional" begin
        @test Nothing <: Optional{Int}
        @test Int <: Optional{Int}
        @test !(String <: Optional{Int})
        @test Optional{Int} == Union{Int, Nothing}
    end

    @testset "@staticmap" begin
        @testset "single pair" begin
            f = @staticmap :a => 1
            @test f(:a) == 1
            @test_throws ArgumentError f(:b)
        end

        @testset "comma-separated" begin
            f = @staticmap :a => 1, :b => 2, :c => 3
            @test f(:a) == 1
            @test f(:b) == 2
            @test f(:c) == 3
            @test_throws ArgumentError f(:d)
        end

        @testset "block form" begin
            f = @staticmap begin
                :x => 10
                :y => 20
            end
            @test f(:x) == 10
            @test f(:y) == 20
        end

        @testset "fallback with _" begin
            f = @staticmap :a => 1, _ => 0
            @test f(:a) == 1
            @test f(:b) == 0
            @test f(:anything) == 0
        end

        @testset "braces grouping" begin
            f = @staticmap {:a, :b, :c} => 1, _ => 2
            @test f(:a) == 1
            @test f(:b) == 1
            @test f(:c) == 1
            @test f(:d) == 2
        end

        @testset "uses === semantics" begin
            f = @staticmap 1 => :int, 1.0 => :float, _ => :other
            @test f(1) == :int
            @test f(1.0) == :float
        end
    end

end
