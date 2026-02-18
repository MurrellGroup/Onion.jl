function watmul(W::DenseArray{T,3}, x::DenseArray{T}) where T
    y = einsum(W, x, einops"s₂ s₁ k, (s₁ k) ... -> (s₂ k) ...")
    return y::typeof(x)
end

const ⨝ = watmul
