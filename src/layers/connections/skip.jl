abstract type AbstractSkipConnection <: AbstractConnection end
function operator end

function (sc::AbstractSkipConnection)(F, x)
    ⊕ = operator(sc)
    return F(x) ⊕ x
end

function (sc::AbstractSkipConnection)(F, x, args...; kws...)
    sc(x -> F(x, args...; kws...), x)
end


struct SkipConnection{Op} <: AbstractSkipConnection
    op::Op
end

operator(sc::SkipConnection) = sc.op


struct ResidualConnection <: AbstractSkipConnection end

operator(::ResidualConnection) = (+)
