struct Untrainable{T} <: Layer; layer::T end
Optimisers.trainable(::Untrainable) = (;)
(layer::Untrainable)(args...; kws...) = layer[](args...; kws...)

Base.getindex(layer::Untrainable) = getfield(layer, :layer)
Base.propertynames(layer::Untrainable) = propertynames(layer[])
Base.getproperty(layer::Untrainable, name::Symbol) = getproperty(layer[], name)
Base.show(io::IO, layer::Untrainable) = print(io, summary(layer))
