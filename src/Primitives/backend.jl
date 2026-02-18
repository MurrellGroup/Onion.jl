using Base.ScopedValues: ScopedValue, with

const CURRENT_BACKEND = ScopedValue{Union{Backend, Nothing}}(nothing)
const GLOBAL_BACKEND = Ref{Union{Backend, Nothing}}(nothing)

backend() = @something(
    CURRENT_BACKEND[],
    GLOBAL_BACKEND[],
    error("no backend set")
)

resolve_backend(b::Backend, ::Primitive) = b
resolve_backend(f::Function, p::Primitive) = f(p)::Backend

backend(rules::Rules) = get(rules, :backend, backend())
backend(rules::Rules, p::Primitive) = resolve_backend(backend(rules), p)

backend!(b::Backend) = (GLOBAL_BACKEND[] = b; nothing)
withbackend(f::Function, b::Backend) = with(f, CURRENT_BACKEND => b)
