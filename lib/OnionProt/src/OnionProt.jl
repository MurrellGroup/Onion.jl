module OnionProt

using Onion
using OnionCore: Layer
using OnionStyle: →, Optional

using ConcreteStructs: @concrete
using Einops: rearrange, einsum, @einops_str
using NNlib

include("pairwise/pairwise.jl")

include("structural/structural.jl")

include("ipa.jl")
export Framemover, IPAblock, CrossFrameIPA, pair_encode

include("dropout.jl")
export training_mode, training_mode!, SharedDropout
export get_dropout_mask, get_dropout_mask_rowise, get_dropout_mask_columnwise

include("combine_projections.jl")

end # module OnionProt
