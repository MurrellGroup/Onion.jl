using .Primitives: @primitive, @impl

@primitive rms_norm
include("norm/rms_norm.jl")

@primitive layer_norm
include("norm/layer_norm.jl")

@primitive softmax
include("softmax.jl")

@primitive attention
include("attention/attention.jl")

@primitive glu_ffn
include("feedforward/glu.jl")

@primitive multihead_ffn
include("feedforward/multihead.jl")
