# ──── AF2/ESMFold TriangleMultiplicativeUpdate ────

"""
    TriangleMultiplicativeUpdate(c_z, c_hidden; outgoing=true)

Triangle multiplicative update. Projects a/b with sigmoid gates,
calls `combine_projections(a, b, outgoing)`, then output gate.
"""
@concrete struct TriangleMultiplicativeUpdate <: Layer
    linear_a_p; linear_a_g; linear_b_p; linear_b_g
    linear_g; linear_z
    norm_in; norm_out
    outgoing::Bool
end

function TriangleMultiplicativeUpdate(c_z::Int, c_hidden::Int; outgoing::Bool=true)
    linear_a_p = Linear(c_z => c_hidden)
    linear_a_g = Linear(c_z => c_hidden)
    linear_b_p = Linear(c_z => c_hidden)
    linear_b_g = Linear(c_z => c_hidden)
    linear_g   = Linear(c_z => c_z)
    linear_z   = Linear(c_hidden => c_z)
    norm_in    = LayerNorm(c_z)
    norm_out   = LayerNorm(c_hidden)
    return TriangleMultiplicativeUpdate(
        linear_a_p, linear_a_g, linear_b_p, linear_b_g,
        linear_g, linear_z, norm_in, norm_out, outgoing,
    )
end

function (m::TriangleMultiplicativeUpdate)(z; mask=nothing)
    z_n = m.norm_in(z)
    a = NNlib.sigmoid.(m.linear_a_g(z_n)) .* m.linear_a_p(z_n)
    b = NNlib.sigmoid.(m.linear_b_g(z_n)) .* m.linear_b_p(z_n)
    if mask !== nothing
        mask_r = reshape(mask, 1, size(mask)...)
        a = a .* mask_r
        b = b .* mask_r
    end
    x = combine_projections(a, b, m.outgoing)
    x = m.linear_z(m.norm_out(x))
    return x .* NNlib.sigmoid.(m.linear_g(z_n))
end

TriangleMultiplicationOutgoing(c_z::Int, c_hidden::Int) =
    TriangleMultiplicativeUpdate(c_z, c_hidden; outgoing=true)

TriangleMultiplicationIncoming(c_z::Int, c_hidden::Int) =
    TriangleMultiplicativeUpdate(c_z, c_hidden; outgoing=false)

# ──── BoltzGen BGTriangleMultiplication ────

"""
    BGTriangleMultiplication(dim; outgoing=true)

BoltzGen-style triangle multiplication with fused projection × sigmoid gate.
"""
@concrete struct BGTriangleMultiplication <: Layer
    norm_in; p_in; g_in; norm_out; p_out; g_out
    outgoing::Bool
end

function BGTriangleMultiplication(dim::Int; outgoing::Bool=true)
    norm_in  = LayerNorm(dim)
    p_in     = Linear(dim => 2dim, bias=false)
    g_in     = Linear(dim => 2dim, bias=false)
    norm_out = LayerNorm(dim)
    p_out    = Linear(dim => dim, bias=false)
    g_out    = Linear(dim => dim, bias=false)
    return BGTriangleMultiplication(norm_in, p_in, g_in, norm_out, p_out, g_out, outgoing)
end

function (l::BGTriangleMultiplication)(x; mask)
    x = l.norm_in(x)
    x_in = x
    x = l.p_in(x) .* NNlib.sigmoid.(l.g_in(x))
    x = x .* reshape(mask, 1, size(mask)...)
    d = size(x_in, 1)
    a = @view x[1:d, :, :, :]
    b = @view x[d+1:2d, :, :, :]
    x = combine_projections(a, b, l.outgoing)
    return l.p_out(l.norm_out(x)) .* NNlib.sigmoid.(l.g_out(x_in))
end

BGTriangleMultiplicationOutgoing(dim::Int) = BGTriangleMultiplication(dim; outgoing=true)
BGTriangleMultiplicationIncoming(dim::Int) = BGTriangleMultiplication(dim; outgoing=false)

# ──── MiniTriangularUpdate ────

"""
    MiniTriangularUpdate(dim)

Compact triangle update: splits into 4 chunks, does both outgoing and incoming
`combine_projections`, then concatenates.
"""
@concrete struct MiniTriangularUpdate <: Layer
    norm_in; p_in; g_in; norm_out; p_out; g_out
end

function MiniTriangularUpdate(dim::Int)
    norm_in  = LayerNorm(dim)
    p_in     = Linear(dim => dim, bias=false)
    g_in     = Linear(dim => dim, bias=false)
    norm_out = LayerNorm(dim ÷ 2)
    p_out    = Linear(dim ÷ 2 => dim, bias=false)
    g_out    = Linear(dim ÷ 2 => dim, bias=false)
    return MiniTriangularUpdate(norm_in, p_in, g_in, norm_out, p_out, g_out)
end

function (l::MiniTriangularUpdate)(x; mask)
    d = size(x, 1)
    x = l.norm_in(x)
    x = l.p_in(x) .* NNlib.sigmoid.(l.g_in(x))
    x = x .* reshape(mask, 1, size(mask)...)

    chunk = d ÷ 4
    a1 = @view x[1:chunk, :, :, :]
    b1 = @view x[chunk+1:2chunk, :, :, :]
    a2 = @view x[2chunk+1:3chunk, :, :, :]
    b2 = @view x[3chunk+1:4chunk, :, :, :]

    x = cat(
        combine_projections(a1, b1, true),
        combine_projections(a2, b2, false);
        dims=1,
    )
    return l.p_out(l.norm_out(x)) .* NNlib.sigmoid.(l.g_out(x))
end
