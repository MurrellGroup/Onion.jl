"""
    SequenceToPair(c_s, c_inner, c_z)

Project sequence state to pairwise via outer product + difference.
Shape: `(C_s, L, B)` → `(C_z, L, L, B)`.
"""
@concrete struct SequenceToPair <: Layer
    norm; proj; o_proj
end

function SequenceToPair(c_s::Int, c_inner::Int, c_z::Int)
    norm   = LayerNorm(c_s)
    proj   = Linear(c_s => 2 * c_inner)
    o_proj = Linear(2 * c_inner => c_z)
    return SequenceToPair(norm, proj, o_proj)
end

function (m::SequenceToPair)(s)
    s = m.proj(m.norm(s))
    d = size(s, 1) ÷ 2
    q = @view s[1:d, :, :]
    k = @view s[d+1:2d, :, :]
    q_exp = rearrange(q, einops"D L B -> D 1 L B")
    k_exp = rearrange(k, einops"D L B -> D L 1 B")
    x = cat(q_exp .* k_exp, q_exp .- k_exp; dims=1)
    return m.o_proj(x)
end

"""
    PairToSequence(c_z, num_heads)

Project pairwise state to per-position attention bias.
Shape: `(C_z, L, L, B)` → `(H, L, L, B)`.
"""
@concrete struct PairToSequence <: Layer
    norm; linear
end

function PairToSequence(c_z::Int, num_heads::Int)
    norm   = LayerNorm(c_z)
    linear = Linear(c_z => num_heads, bias=false)
    return PairToSequence(norm, linear)
end

(m::PairToSequence)(z) = m.linear(m.norm(z))

"""
    ResidueMLP(dim, inner_dim; dropout=0)

Two-layer MLP with ReLU and residual connection.
"""
@concrete struct ResidueMLP <: Layer
    norm; fc1; fc2; dropout
end

function ResidueMLP(dim::Int, inner_dim::Int; dropout::Real=0)
    norm = LayerNorm(dim)
    fc1  = Linear(dim => inner_dim)
    fc2  = Linear(inner_dim => dim)
    drop = SharedDropout(dropout, 3)
    return ResidueMLP(norm, fc1, fc2, drop)
end

function (m::ResidueMLP)(x)
    y = max.(m.fc1(m.norm(x)), 0f0)
    return x .+ m.dropout(m.fc2(y))
end
