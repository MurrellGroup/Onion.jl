using Flux: _paramtype

bf16(m) = _paramtype(BFloat16, m)
