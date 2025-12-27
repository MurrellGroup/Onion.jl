# Ops

`Onion.Ops` binds optimized primitives for heavier layers (like attention, softmax, and normalization),
ensuring that the code runs on any device, since e.g. [ONIONop.jl](https://github.com/MurrellGroup/ONIONop.jl) doesn't
doesn't support running on CPU.
