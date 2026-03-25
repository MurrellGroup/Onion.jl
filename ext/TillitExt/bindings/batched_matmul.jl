function Onion.batched_matmul!(::TillitBackend,
    C::AbstractArray{T,3},
    A::AbstractArray{T,3}, B::AbstractArray{T,3};
    kws...
) where T
    function verify()
        C_ref = Onion.batched_matmul(DefaultBackend(), A, B)
        function iscorrect()
            isapprox(C, C_ref, atol=1e-1, rtol=1e-1)
        end
    end
    Tillit.batched_matmul!(A, B, C; verify, kws...)
    return C
end

function CRC.rrule(::typeof(Onion.batched_matmul!), backend::TillitBackend,
    C::AbstractArray{T,3},
    A::AbstractArray{T,3}, B::AbstractArray{T,3}
) where T
    C = Onion.batched_matmul!(backend, C, A, B)
    function batched_matmul_pullback(C̄)
        C̄ = unthunk(C̄)
        Ā, B̄ = similar(A), similar(B)
        function verify_a()
            _, pb = Zygote.pullback(A→Float32, B→Float32) do A, B
                Onion.batched_matmul(DefaultBackend(), A, B)
            end
            Ā_ref, = pb(C̄→Float32)
            function iscorrect()
                isapprox(Ā, Ā_ref, atol=1e-1, rtol=1e-1)
            end
        end
        function verify_b()
            _, pb = Zygote.pullback(A→Float32, B→Float32) do A, B
                Onion.batched_matmul(DefaultBackend(), A, B)
            end
            _, B̄_ref = pb(C̄→Float32)
            function iscorrect()
                isapprox(B̄, B̄_ref, atol=1e-1, rtol=1e-1)
            end
        end
        Tillit.∇batched_matmul!(Ā, B̄, C̄, A, B; verify_a, verify_b)
        return NoTangent(), NoTangent(), NoTangent(), Ā, B̄
    end
    return C, batched_matmul_pullback
end
