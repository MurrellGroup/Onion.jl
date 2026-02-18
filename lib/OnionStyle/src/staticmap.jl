macro staticmap(args...)
    length(args) == 1 || error("@staticmap expects a block or comma-separated pairs")
    arg = args[1]
    pairs = if arg isa Expr && arg.head === :block
        filter(ex -> !(ex isa LineNumberNode), arg.args)
    elseif arg isa Expr && arg.head === :tuple
        arg.args
    elseif arg isa Expr && arg.head === :call && arg.args[1] === :(=>)
        [arg]
    else
        error("@staticmap expects a block or comma-separated pairs")
    end
    checks = Expr[]
    fallback = nothing
    for ex in pairs
        ex isa Expr && ex.head === :call && ex.args[1] === :(=>) ||
            error("@staticmap: expected `key => value`, got `$ex`")
        f, b = ex.args[2], ex.args[3]
        if f === :_
            fallback = b
        elseif f isa Expr && f.head === :braces
            cond = foldl((a, fi) -> :(($a) || x === $fi), f.args[2:end]; init = :(x === $(f.args[1])))
            push!(checks, :($cond && return $b))
        else
            push!(checks, :(x === $f && return $b))
        end
    end
    default = isnothing(fallback) ?
        :(throw(ArgumentError("no mapping for $(typeof(x))"))) :
        :(return $fallback)
    fname = gensym(:staticmap)
    esc(:(function $fname(x)
        $(checks...)
        $default
    end))
end
