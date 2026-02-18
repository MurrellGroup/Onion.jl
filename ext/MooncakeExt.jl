module MooncakeExt

using Onion: Primitive, cuTileBackend
using Mooncake: @is_primitive, DefaultCtx

@is_primitive DefaultCtx Tuple{Primitive, cuTileBackend, Vararg{Any}}

end
