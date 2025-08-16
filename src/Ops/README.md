# Ops

`Onion.Ops` binds optimized primitives for heavier layers (like softmax, attention, and normalization),
ensuring that the code runs on any device, since e.g. [NNop.jl](https://github.com/pxl-th/NNop.jl) doesn't
doesn't support running on CPU.
