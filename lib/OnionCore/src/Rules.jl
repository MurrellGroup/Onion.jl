"""
    Rules(; kws...)

A typed key-value container for propagating configuration through layer calls.
Wraps a `NamedTuple`, so field names and types are part of the type — enabling
static dispatch on the contents.

    rules = Rules(backend=TileBackend(), precision=Float16)
    rules.backend  # TileBackend()

Rules can be merged, with later values overriding earlier ones:

    merge(rules, Rules(precision=Float32))
"""
struct Rules{names,T<:Tuple}
    fields::NamedTuple{names,T}
end

function Rules(; kws...)
    fields = NamedTuple(kws)
    :fields in keys(fields) && error("field name 'fields' not permitted")
    return Rules(fields)
end

Base.NamedTuple(r::Rules) = r.fields
Base.keys(::Rules{names}) where names = names
Base.values(r::Rules) = values(NamedTuple(r))
Base.eltype(r::Rules) = Pair{Symbol,eltype(NamedTuple(r))}
Base.length(r::Rules) = length(keys(r))
Base.getindex(r::Rules, i::Int) = keys(r)[i] => values(r)[i]
Base.getindex(r::Rules, name::Symbol) = getproperty(r, name)
Base.merge(a::Rules, b::Rules) = Rules(merge(NamedTuple(a), NamedTuple(b)))

function Base.get(r::Rules, name::Symbol, default)
    return hasproperty(r, name) ? getproperty(r, name) : default
end

function Base.iterate(r::Rules, i::Int=1)
    return i <= length(r) ? (r[i], i+1) : nothing
end

function Base.propertynames(r::Rules, private::Bool=false)
    return private ? (keys(r)..., :fields) : keys(r)
end

function Base.getproperty(x::Rules, name::Symbol)
    fields = getfield(x, :fields)
    return name === :fields ? fields : getfield(fields, name)
end

function Base.show(_io::IO, rules::Rules)
    io = IOContext(_io, :limit => true, :compact => true)
    print(io, Rules, '(')
    for name in keys(rules)
        print(io, name, " = ", getproperty(rules, name))
        name != last(keys(rules)) && print(io, ", ")
    end
    print(io, ')')
end
