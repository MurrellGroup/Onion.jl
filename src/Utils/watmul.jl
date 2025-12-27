function watmul(W::DenseArray{T,3}, x::DenseArray{T}) where T
    y = einsum(W, x, einops"s₂ s₁ k, (s₁ k) ... -> (s₂ k) ...")
    return y::typeof(x)
end

const ⨝ = watmul

function watmul_mix(
    W::DenseArray{T,3}, x::DenseArray{T};
    W_in::DenseMatrix{T}
) where T
    y = einsum(x, W_in, einops"(s₁ k₁) ..., k₁ k₂ -> (s₁ k₂) ...")
    z = einsum(W, y, einops"s₂ s₁ k₂, (s₁ k₂) ... -> (s₂ k₂) ...")
    return z::typeof(x)
end
