function Onion._fused_add_rms_norm(::cuTileBackend,
    residual::AbstractArray{T}, x::AbstractArray{T}, w::AbstractVector;
    eps, offset,
) where T
    r2d = ndims(residual) == 1 ? reshape(residual, :, 1) : residual
    x2d = ndims(x) == 1 ? reshape(x, :, 1) : x

    NewX = similar(r2d)
    Normed = similar(x2d)
    tol = T === Float32 ? 1e-3 : 1e-1

    function verify()
        new_ref, normed_ref = Onion._fused_add_rms_norm(DefaultBackend(),
            Array(r2d), Array(x2d), Array(w); eps, offset)
        function iscorrect()
            isapprox(Float32.(Array(NewX)), Float32.(new_ref); atol=tol, rtol=tol) &&
            isapprox(Float32.(Array(Normed)), Float32.(normed_ref); atol=tol, rtol=tol)
        end
    end

    fused_add_rms_norm_step!(NewX, Normed, r2d, x2d, w;
        eps=Float32(eps), offset=Float32(offset))

    if ndims(residual) == 1
        return vec(NewX), vec(Normed)
    end
    return NewX, Normed
end
