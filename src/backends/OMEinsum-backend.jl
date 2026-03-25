using OMEinsum: OMEinsum

"""
    einsum_cutensor_backend!(enabled::Bool)

Change the backend of [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl).

Default implementations of Onion primitives may use
[Einops.jl](https://github.com/MurrellGroup/Einops.jl)'s
`einsum`, which uses OMEinsum.

For NVIDIA GPUs, you may import `cuTENSOR` and
use `CuTensorBackend` in OMEinsum with:

    einsum_cutensor_backend!(true)
"""
function einsum_cutensor_backend!(enabled::Bool)
    if enabled
        OMEinsum.set_einsum_backend!(OMEinsum.CuTensorBackend())
    else
        OMEinsum.set_einsum_backend!(OMEinsum.DefaultBackend())
    end
    return nothing
end
