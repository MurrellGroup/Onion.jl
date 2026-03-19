function _q_mask(q_lengths, O)
    isnothing(q_lengths) && return true
    reshape(1:size(O,2), 1, :, 1, 1) .<= reshape(q_lengths, 1, 1, 1, :)
end

function Onion.attention!(::cuTileBackend,
    O::AbstractArray{T,4},
    Q::AbstractArray{T,4}, K::AbstractArray{T,4}, V::AbstractArray{T,4};
    q_lengths = nothing,
    pair = nothing, kws...
) where T
    function verify()
        O_ref = Onion.attention(DefaultBackend(), Q→Float32, K→Float32, V→Float32; pair=B→Float32, kws...)
        qm = _q_mask(q_lengths, O)
        function iscorrect()
            isapprox(O .* qm, O_ref .* qm, atol=1e-1, rtol=1e-1)
        end
    end
    flash_attention!(O, Q, K, V, pair; q_lengths, kws...)
    return O
end

function CRC.rrule(::typeof(Onion.attention!), ::cuTileBackend,
    O::AbstractArray,
    Q::AbstractArray, K::AbstractArray, V::AbstractArray, B;
    q_lengths = nothing, kws...
)
    M = similar(Q, Float32, size(Q)[2:4])
    L = similar(Q, Float32, size(Q)[2:4])
    flash_attention!(O, Q, K, V, B; M, L, kws...)
    function flash_attention_pullback(Ō)
        Q̄, K̄, V̄ = similar.((Q, K, V))
        B̄ = isnothing(B) ? nothing : fill!(similar(B), 0)
        function verify()
            qm = _q_mask(q_lengths, Q)
            Qm, Ōm = (Q .* qm)→Float32, (Ō .* qm)→Float32
            _, pb = Zygote.pullback(Qm, K→Float32, V→Float32, B→Float32) do Q, K, V, B
                Onion.attention(DefaultBackend(), Q, K, V; pair=B, kws...)
            end
            Q̄_ref, K̄_ref, V̄_ref, B̄_ref = pb(Ōm)
            function iscorrect()
                isapprox(Q̄ .* qm, Q̄_ref .* qm, atol=1e-1, rtol=1e-1) &&
                isapprox(K̄, K̄_ref, atol=1e-1, rtol=1e-1) &&
                isapprox(V̄, V̄_ref, atol=1e-1, rtol=1e-1) &&
                (isnothing(B) || isapprox(B̄, B̄_ref, atol=1e-1, rtol=1e-1))
            end
        end
        ∇flash_attention!(
            Q̄, K̄, V̄, B̄, unthunk(Ō),
            Q, K, V, B, O, M, L;
            verify, q_lengths, kws...)
        return NoTangent(), Q̄, K̄, V̄, @something B̄ NoTangent()
    end
    return O, flash_attention_pullback
end
