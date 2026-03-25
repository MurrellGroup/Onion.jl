"""
    BackendMap(resolve)

A backend that resolves to a concrete backend per-primitive.
Wraps a callable `resolve(::Primitive) -> Backend`.

    BackendMap(@staticmap begin
        attention => TillitBackend()
        _ => DefaultBackend()
    end)
"""
struct BackendMap{F}
    resolve::F
end
(b::BackendMap)(p::Primitive) = b.resolve(p)
(p::Primitive)(b::BackendMap, args...; kws...) = p(b(p), args...; kws...)

using Base.ScopedValues: ScopedValue, with

const SCOPED_BACKEND = ScopedValue{Union{Backend,BackendMap,Nothing}}(nothing)
const GLOBAL_BACKEND = Ref{Union{Backend,BackendMap,Nothing}}(nothing)

backend() = @something(
    SCOPED_BACKEND[],
    GLOBAL_BACKEND[],
    error("no backend set")
)

backend(rules::Rules) = get(() -> backend(), rules, :backend)

withbackend(f::Function, b::Union{Backend,BackendMap}) = with(f, SCOPED_BACKEND => b)

backend_available(::Union{Backend,BackendMap}) = (true, "")

function backend!(b::Union{Backend,BackendMap})
    available, msg = backend_available(b)
    available || @warn "$(nameof(typeof(b))): $msg."
    GLOBAL_BACKEND[] = b
    return nothing
end

struct DefaultBackend <: Backend end

struct NNkernelsBackend <: Backend end
const _NNKERNELS_AVAILABLE = Ref{Bool}(false)
function backend_available(::NNkernelsBackend)
    if _NNKERNELS_AVAILABLE[]
        (true, "")
    else
        (false, "NNkernels.jl not loaded")
    end
end

struct TillitBackend <: Backend end
const _TILLIT_AVAILABLE = Ref{Bool}(false)
function backend_available(::TillitBackend)
    if _TILLIT_AVAILABLE[]
        (true, "")
    else
        (false, "Tillit.jl not loaded")
    end
end

# doesn't use our backend system,
# but DefaultBackend sometimes uses `Einops.einsum` for convenience,
# and this controls whether to use cuTENSOR
include("OMEinsum-backend.jl")
public einsum_cutensor_backend!
