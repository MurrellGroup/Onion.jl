macro primitive(name::Symbol)
    T = Symbol(:primitive_, name)
    esc(quote
        struct $T <: $Primitive end
        const $name = $T()
        $(Expr(:public, name))
    end)
end

macro impl(backend, func_def)
    func_def isa Expr && func_def.head in (:function, :(=)) ||
        error("@impl expects a function definition")

    call = func_def.args[1]

    # unwrap `where` clauses
    inner_call = call
    while inner_call isa Expr && inner_call.head == :where
        inner_call = inner_call.args[1]
    end

    inner_call isa Expr && inner_call.head == :call ||
        error("@impl expects a named function definition")

    # insert ::B after name, after :parameters if present
    insert_pos = 2
    if length(inner_call.args) >= 2 &&
        inner_call.args[2] isa Expr &&
        inner_call.args[2].head == :parameters
        insert_pos = 3
    end
    insert!(inner_call.args, insert_pos, :(::$backend))

    return esc(func_def)
end
