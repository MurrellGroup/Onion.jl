using Rewrap: Keep, Split, (..)

"""
    @asmatrix f(\$x, args...)
    @asmatrix f(\$x, args...)[i]

Flatten the `\$`-marked argument's dims 2:end into a single matrix,
call the function, and reshape the result back to the original shape.

Uses Rewrap's `Keep` and `Split` to preserve wrapper types like `PermutedDimsArray`.

Supports indexing into the result before reshaping, useful for functions
that return tuples (e.g. `layer_norm` returning `(Y, Mean, Rstd)`):

```julia
@asmatrix layer_norm(\$x, w, b; eps)[1]
```

# Example

```julia
@asmatrix online_softmax(\$x)
```

expands to roughly:

```julia
let orig = x
    flat = reshape(orig, Keep(), :)
    result = online_softmax(flat)
    reshape(result, Keep(), Split(.., size(orig)[2:end]))
end
```
"""
macro asmatrix(expr)
    # Handle f($x, ...)[i] — indexing into the call result
    if Meta.isexpr(expr, :ref) && Meta.isexpr(expr.args[1], :call)
        call = expr.args[1]
        indices = expr.args[2:end]
    elseif Meta.isexpr(expr, :call)
        call = expr
        indices = nothing
    else
        error("@asmatrix: expected a function call or indexed function call")
    end

    orig_expr = nothing
    dollar_idx = nothing

    for (i, arg) in enumerate(call.args)
        if Meta.isexpr(arg, :$)
            dollar_idx === nothing || error("@asmatrix: expected exactly one \$-marked argument")
            dollar_idx = i
            orig_expr = arg.args[1]
        end
    end

    orig_expr === nothing && error("@asmatrix: expected one \$-marked argument")

    orig = gensym("orig")
    flat = gensym("flat")
    result = gensym("result")

    new_args = Any[]
    for (i, arg) in enumerate(call.args)
        if i == dollar_idx
            push!(new_args, flat)
        else
            push!(new_args, esc(arg))
        end
    end
    new_call = Expr(:call, new_args...)

    # Apply indexing before reshape if present
    if indices !== nothing
        new_call = Expr(:ref, new_call, map(esc, indices)...)
    end

    quote
        $orig = $(esc(orig_expr))
        $flat = reshape($orig, Keep(), :)
        $result = $new_call
        reshape($result, Keep(), Split(.., size($orig)[2:end]))
    end
end
