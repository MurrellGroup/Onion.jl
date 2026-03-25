module TillitExt

using Tillit
using Onion: _TILLIT_AVAILABLE

function __init__()
    _TILLIT_AVAILABLE[] = true
end

include("bindings/bindings.jl")

end
