# BoltzGen triangle attention uses the protein TriangleAttention (defined in protein/triangular.jl)
# which routes through OFMultiheadAttention → flash_attention_forward dispatch hooks for GPU.
# These aliases preserve the BoltzGen API.

const TriangleAttentionStartingNode = TriangleAttention

function TriangleAttentionEndingNode(
    c_in::Int,
    c_hidden::Int,
    no_heads::Int;
    inf::Real=1f9,
)
    return TriangleAttention(c_in, c_hidden, no_heads; starting=false, inf=inf)
end
