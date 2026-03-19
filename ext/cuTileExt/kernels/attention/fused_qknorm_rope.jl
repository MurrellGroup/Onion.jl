# Fused QK-norm + partial RoPE. Grid: (num_heads, B) per Q/K.
# Each block normalizes one head then applies rotary to first rotary_dim dims.
# Q and K are (head_dim, 1, num_heads, B).

function fused_qknorm_rope_fwd(
    Q::TileArray4, K::TileArray4,
    QW::TileVector, KW::TileVector,
    Cos::TileVector, Sin::TileVector,
    D::Int, HALF_ROT::Int, PASS_DIM::Int,
    eps::Float32, Offset::Float32,
)
    padding_mode = ct.PaddingMode.Zero
    h, b = ct.bid(1), ct.bid(2)

    q = ct.load(Q, (1, 1, h, b), (D,); padding_mode) → Float32
    k = ct.load(K, (1, 1, h, b), (D,); padding_mode) → Float32
    qw = ct.load(QW, (1,), (D,); padding_mode) → Float32
    kw = ct.load(KW, (1,), (D,); padding_mode) → Float32

    q_rstd = 1 / √(sum(q .* q) / D + eps)
    q = q .* q_rstd .* (qw .+ Offset)
    k_rstd = 1 / √(sum(k .* k) / D + eps)
    k = k .* k_rstd .* (kw .+ Offset)

    # XXX: redundant?
    ct.store(Q, (1, 1, h, b), q → eltype(Q))
    ct.store(K, (1, 1, h, b), k → eltype(K))

    # Partial RoPE: rotate first 2*HALF_ROT dims in-place
    cos_v = ct.load(Cos, (1,), (HALF_ROT,)) → Float32
    sin_v = ct.load(Sin, (1,), (HALF_ROT,)) → Float32

    q1 = ct.extract(q, Val((1,)), Val((HALF_ROT,)))
    q2 = ct.extract(q, Val((2,)), Val((HALF_ROT,)))
    k1 = ct.extract(k, Val((1,)), Val((HALF_ROT,)))
    k2 = ct.extract(k, Val((2,)), Val((HALF_ROT,)))

    # Overwrite first 2*HALF_ROT dims with rotated values
    q_rot = ct.cat((q1 .* cos_v .- q2 .* sin_v, q2 .* cos_v .+ q1 .* sin_v), 1)
    k_rot = ct.cat((k1 .* cos_v .- k2 .* sin_v, k2 .* cos_v .+ k1 .* sin_v), 1)

    ct.store(Q, (1, 1, h, b), q_rot → eltype(Q))  # stores to first 2*HALF_ROT positions
    ct.store(K, (1, 1, h, b), k_rot → eltype(K))

    return
end

function fused_qknorm_rope_step!(Q, K, QW, KW, Cos, Sin;
    rotary_dim, eps, offset, verify=nothing)
    D, _, H, B = size(Q)
    half_rot = rotary_dim ÷ 2
    pass_dim = D - rotary_dim
    key = (eltype(Q), D, rotary_dim)

    autotune_launch(fused_qknorm_rope_fwd,
        CartesianSpace(),
        cfg -> (H, B),
        cfg -> (
            Q, K, QW, KW, Cos, Sin,
            Constant(D), Constant(half_rot), Constant(pass_dim),
            Constant(Float32(eps)), Constant(Float32(offset)),
        );
        key, verify,
    )
end
