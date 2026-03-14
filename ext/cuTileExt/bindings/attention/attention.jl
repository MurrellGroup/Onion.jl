function _q_mask(q_lengths, O)
    isnothing(q_lengths) && return true
    reshape(1:size(O,2), 1, :, 1, 1) .<= reshape(q_lengths, 1, 1, 1, :)
end

function _flash_attention(Q, K, V, B; q_lengths=nothing, kws...)
    O = similar(Q, size(V,1), size(Q)[2:end]...)
    verify = () -> let
        O_ref = Onion.attention(DefaultBackend(), Q→Float32, K→Float32, V→Float32; pair=B→Float32, kws...)
        qm = _q_mask(q_lengths, O)
        () -> isapprox(O .* qm, O_ref .* qm, rtol=1e-1)
    end
    cache = flash_attention!(O, Q, K, V, B; verify, q_lengths, kws...)
    return O, cache
end

# TODO: make `pair` positional for fewer functions?
function flash_attention(args...; kws...)
    O, _ = _flash_attention(args...; kws...)
    return O
end

function ∇flash_attention(Ō, Q, K, V, B, O, (M, L); q_lengths=nothing, kws...)
    Q̄, K̄, V̄ = similar.((Q, K, V))
    B̄ = isnothing(B) ? nothing : fill!(similar(B), 0)
    verify = () -> let
        qm = _q_mask(q_lengths, Q)
        Qm, Ōm = (Q .* qm)→Float32, (Ō .* qm)→Float32
        _, pb = Zygote.pullback(Qm, K→Float32, V→Float32, B→Float32) do Q, K, V, B
            Onion.attention(DefaultBackend(), Q, K, V; pair=B, kws...)
        end
        Q̄_ref, K̄_ref, V̄_ref, B̄_ref = pb(Ōm)
        () -> isapprox(Q̄ .* qm, Q̄_ref .* qm, rtol=1e-1) &&
              isapprox(K̄, K̄_ref, rtol=1e-1) &&
              isapprox(V̄, V̄_ref, rtol=1e-1) &&
              (isnothing(B) || isapprox(B̄, B̄_ref, rtol=1e-1))
    end
    ∇flash_attention!(
        Q̄, K̄, V̄, B̄, Ō,
        Q, K, V, B, O,
        M, L; verify, q_lengths, kws...)
    return Q̄, K̄, V̄, B̄
end

function CRC.rrule(::typeof(flash_attention),
    Q::AbstractArray, K::AbstractArray, V::AbstractArray, B;
    kws...
)
    O, cache = _flash_attention(Q, K, V, B; kws...)
    function flash_attention_pullback(Ō)
        Q̄, K̄, V̄, B̄ = ∇flash_attention(unthunk(Ō), Q, K, V, B, O, cache; kws...)
        return NoTangent(), Q̄, K̄, V̄, @something B̄ NoTangent()
    end
    return O, flash_attention_pullback
end

function Onion._attention(::cuTileBackend,
    Q::AbstractArray, K::AbstractArray, V::AbstractArray;
    pair = nothing, kws...
)
    O = flash_attention(Q, K, V, pair; kws...)
    return O
end
