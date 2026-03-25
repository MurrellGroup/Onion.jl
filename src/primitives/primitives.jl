"""
    get_buffers(primitive, backend, args...; kws...) -> NamedTuple

Allocate output buffers for a primitive. Returns a NamedTuple of pre-allocated
arrays sized for the given inputs. Use with the mutating variant:

    bufs = get_buffers(linear, backend, x, W, b)
    linear!(backend, bufs.y, x, W, b)

For CUDA graphs: call `get_buffers` once at setup, reuse buffers across replays.
"""
function get_buffers end
public get_buffers

include("linear.jl")
include("softmax.jl")
include("norm/norm.jl")
include("attention/attention.jl")
include("feedforward/feedforward.jl")
include("batched_matmul.jl")
include("recurrent/recurrent.jl")
include("rotary.jl")
