function decode end

# decode is called directly and doesnt
# inherit Rules from some (::Layer)(args...)
decode(layer::Layer, args...; kws...) = decode(layer, Rules(), args...; kws...)

decode(layer::Layer, r::Rules, args...; kws...) =
    throw(MethodError(decode, (layer, r, args...)))
