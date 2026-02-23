module cuTileExt

import OnionStyle: _to
using cuTile: Tile

_to(::Type{T}, x::Tile{<:AbstractFloat}) where T = T.(x)

end
