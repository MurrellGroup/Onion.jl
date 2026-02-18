using OnionCore
using OnionCore: Backend, LayerStyle, EagerStyle, apply, apply_with, forward
using Test

@testset "OnionCore" begin

    @testset "Rules" begin
        @testset "construction" begin
            r = Rules()
            @test length(r) == 0
            @test keys(r) == ()

            r = Rules(a=1, b="hello")
            @test length(r) == 2
            @test keys(r) == (:a, :b)
            @test values(r) == (1, "hello")
        end

        @testset "field name 'fields' is forbidden" begin
            @test_throws ErrorException Rules(fields=1)
        end

        @testset "property access" begin
            r = Rules(x=42, y=:foo)
            @test r.x == 42
            @test r.y == :foo
            @test propertynames(r) == (:x, :y)
            @test propertynames(r, true) == (:x, :y, :fields)
        end

        @testset "getindex" begin
            r = Rules(a=1, b=2)
            @test r[1] == (:a => 1)
            @test r[2] == (:b => 2)
            @test r[:a] == 1
            @test r[:b] == 2
        end

        @testset "get with default" begin
            r = Rules(a=1)
            @test get(r, :a, 0) == 1
            @test get(r, :b, 0) == 0
        end

        @testset "merge" begin
            a = Rules(x=1, y=2)
            b = Rules(y=3, z=4)
            m = merge(a, b)
            @test m.x == 1
            @test m.y == 3
            @test m.z == 4
        end

        @testset "iterate" begin
            r = Rules(a=1, b=2)
            pairs = collect(r)
            @test pairs == [:a => 1, :b => 2]
        end

        @testset "NamedTuple conversion" begin
            r = Rules(a=1, b=2)
            nt = NamedTuple(r)
            @test nt === (a=1, b=2)
        end

        @testset "eltype" begin
            r = Rules(a=1, b=2)
            @test eltype(r) == Pair{Symbol, Int}
        end

        @testset "show" begin
            r = Rules(a=1, b="hi")
            s = sprint(show, r)
            @test contains(s, "Rules")
            @test contains(s, "a = 1")
            @test contains(s, "b = hi")
        end

        @testset "type stability" begin
            r = Rules(a=1, b=2.0)
            @test typeof(r) == Rules{(:a, :b), Tuple{Int, Float64}}
        end
    end

    @testset "Layer" begin
        struct TestLayer <: Layer
            alpha::Float64
            beta::Float64
        end

        OnionCore.forward(l::TestLayer, x) = l.alpha * x + l.beta

        @testset "callable" begin
            l = TestLayer(2.0, 3.0)
            @test l(5.0) == 13.0
        end

        @testset "callable with Rules" begin
            l = TestLayer(2.0, 3.0)
            r = Rules(precision=Float32)
            @test l(r, 5.0) == 13.0
        end

        @testset "forward strips Rules by default" begin
            l = TestLayer(2.0, 3.0)
            @test forward(l, Rules(), 5.0) == 13.0
        end

        @testset "missing forward errors" begin
            struct EmptyLayer <: Layer end
            @test_throws ErrorException EmptyLayer()(1.0)
        end

        @testset "LayerStyle defaults to EagerStyle" begin
            @test LayerStyle(TestLayer) isa EagerStyle
            @test LayerStyle(TestLayer(1.0, 2.0)) isa EagerStyle
        end

        @testset "unicode properties" begin
            l = TestLayer(2.0, 3.0)
            @test l.α == 2.0
            @test l.β == 3.0
            @test l.alpha == 2.0
            @test l.beta == 3.0
        end

        @testset "unicode property not found" begin
            l = TestLayer(2.0, 3.0)
            @test_throws Exception l.γ
        end
    end

end
