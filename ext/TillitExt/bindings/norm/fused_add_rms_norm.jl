function Onion.fused_add_rms_norm!(::TillitBackend,
    new_x::AbstractMatrix{T}, normed::AbstractMatrix{T},
    residual::AbstractMatrix{T}, x::AbstractMatrix{T}, w::AbstractVector;
    eps, offset,
) where T

    function verify()
        tol = T === Float32 ? 1e-3 : 1e-1
        new_ref, normed_ref = Onion.fused_add_rms_norm(DefaultBackend(),
            Array(r2d), Array(x2d), Array(w); eps, offset)
        function iscorrect()
            isapprox(Float32.(Array(NewX)), Float32.(new_ref); atol=tol, rtol=tol) &&
            isapprox(Float32.(Array(Normed)), Float32.(normed_ref); atol=tol, rtol=tol)
        end
    end

    Tillit.fused_add_rms_norm!(new_x, normed, residual, x, w;
        eps=Float32(eps), offset=Float32(offset))

    return new_x, normed
end
