@testset "DyT.jl" begin
    dyt = DyT(256)
    x = randn(Float32, 256, 2)
    y = dyt(x)
    @test size(y) == size(x)
end