abstract type AbstractSkipConnections <: AbstractConnections end
function operator end

function (sc::AbstractSkipConnections)(F, x)
    ⊕ = operator(sc)
    return F(x) .⊕ x
end

function (sc::AbstractSkipConnections)(F, x, args...; kws...)
    sc(x -> F(x, args...; kws...), x)
end


struct SkipConnections{Op} <: AbstractSkipConnections
    op::Op
end

operator(sc::SkipConnections) = sc.op


struct ResidualConnections <: AbstractSkipConnections end

operator(::ResidualConnections) = (+)
