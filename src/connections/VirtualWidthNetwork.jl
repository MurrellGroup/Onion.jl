@concrete struct VirtualWidthNetwork <: Layer
    ghc <: GHC
    layer
end

VirtualWidthNetwork(layer, n::Int, m::Int) = VirtualWidthNetwork(GHC(n, m), layer)

(vwn::VirtualWidthNetwork)(x) = vwn.ghc(vwn.layer, x)
