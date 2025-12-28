@concrete struct Composed <: Layer
    inner
    outer
end

function ((; inner, outer)::Composed)(args...; kws...)
    return outer(inner(args...; kws...))
end
