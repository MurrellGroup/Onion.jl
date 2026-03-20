# Fallback: pass through to original call
inplace(f, args...; kws...) = f(args...; kws...)

function inplace_walk(ex)
    ex isa Expr || return ex
    if ex.head === :call
        f = ex.args[1]
        args = map(inplace_walk, ex.args[2:end])
        return Expr(:call, GlobalRef(OnionStyle, :inplace), f, args...)
    elseif ex.head === :(.)
        # dot-call: f.(args...) is Expr(:(.), f, Expr(:tuple, args...))
        if length(ex.args) == 2 && ex.args[2] isa Expr && ex.args[2].head === :tuple
            f = ex.args[1]
            inner = Expr(:tuple, map(inplace_walk, ex.args[2].args)...)
            return Expr(:(.), GlobalRef(OnionStyle, :inplace), Expr(:tuple, f, inner.args...))
        end
        return ex
    elseif ex.head === :kw
        # keyword argument: recurse into value only
        return Expr(:kw, ex.args[1], inplace_walk(ex.args[2]))
    else
        # blocks, assignments, etc: recurse into all subexpressions
        return Expr(ex.head, map(inplace_walk, ex.args)...)
    end
end

macro !(ex)
    esc(inplace_walk(ex))
end
