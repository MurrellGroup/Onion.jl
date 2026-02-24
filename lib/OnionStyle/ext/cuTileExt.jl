module cuTileExt

import OnionStyle: →
using cuTile: Tile

x::Tile{<:AbstractFloat} → T::Type = T.(x)

end
