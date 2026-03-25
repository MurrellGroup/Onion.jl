module NNkernelsExt

using NNkernels
using Onion: _NNKERNELS_AVAILABLE

function __init__()
    _NNKERNELS_AVAILABLE[] = true
end

include("bindings/bindings.jl")

end
