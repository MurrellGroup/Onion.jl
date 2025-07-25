import BatchedTransformations as BT

batched_pairs(operator, a, b) = operator.(reshape(a, 1, :, size(a,2)),reshape(b, :, 1, size(b,2)))

function pair_encode(resinds, chainids)
    chain_diffs = Float32.(batched_pairs(==, chainids, chainids))
    num_diffs = Float32.(batched_pairs(-, resinds, resinds))
    decay_num_diffs = (sign.(num_diffs) ./ (1 .+ abs.(num_diffs) ./ 5)) .* chain_diffs
    return vcat(reshape(decay_num_diffs, 1, size(decay_num_diffs)...), reshape(chain_diffs, 1, size(chain_diffs)...))
end

function pairwise_sqeuclidean(x,y)
    A_sqnorms = sum(abs2, x, dims=2)
    B_sqnorms = sum(abs2, y, dims=1)
    AB_dots = batched_mul(x,y)
    return A_sqnorms .- 2 .* AB_dots .+ B_sqnorms
end
#d = pairwise_sqeuclidean(permutedims(p, (2,1,3)), p)

"""
    Framemover(dim::Int; init_gain = 0.1f0)

Differentiable rigid body updates (AF2-style).
"""
struct Framemover{A,B}
    loc_decode::A
    rot_decode::B
end

Flux.@layer Framemover

function Framemover(dim::Int; init_gain = 0.1f0)
    loc_decode = Dense(dim => 3, bias = false, init = Flux.glorot_uniform(gain = init_gain))
    rot_decode = Dense(dim => 3, bias = false, init = Flux.glorot_uniform(gain = init_gain))
    return Framemover(loc_decode, rot_decode)
end

bcd2rot(bcds) = convert(Rotation, BT.QuaternionRotation(imaginary_to_quaternion_rotations(bcds)))

function (fm::Framemover)(frames, x; t = 0)
    bcds = fm.rot_decode(x) .* glut(1 .- t, ndims(x), 0)
    loc_change = glut(fm.loc_decode(x)  .* glut(1 .- t, ndims(x), 0), ndims(x)+1, 1)
    return frames ∘ (Translation(loc_change) ∘ bcd2rot(bcds))
end


"""
    IPAblock(dim::Int, ipa; ln1 = Flux.LayerNorm(dim), ln2 = Flux.LayerNorm(dim), ff = StarGLU(dim, 3dim))

For use with Invariant Point Attention, either from InvariantPointAttention.jl or MessagePassingIPA.jl.
If `ipablock.ipa` is from InvariantPointAttention.jl, then call `ipablock(frames, x; pair_feats = nothing, cond = nothing, mask = 0, kwargs...)`
If `ipablock.ipa` is from MessagePassingIPA.jl, then call `ipablock(g, frames, x, pair_feats; cond = nothing)`
Pass in `cond` if you're using eg. `AdaLN` that takes a second argument. 
"""
struct IPAblock{A,B,C,D}
    ln1::A
    ipa::B
    ln2::C
    ff::D
end

Flux.@layer IPAblock

IPAblock(dim::Int, ipa; ln1 = Flux.LayerNorm(dim), ln2 = Flux.LayerNorm(dim), ff = StarGLU(dim, 3dim)) = IPAblock(ln1, ipa, ln2, ff)

lncall(ln, x, cond) = ln(x, cond)
lncall(ln, x, cond::Nothing) = ln(x)

#InvariantPointAttention.jl:
function (ipa_block::IPAblock)(frames::Rigid, x; pair_feats = nothing, cond = nothing, mask = 0, kwargs...)
    T = values(linear(frames)), values(translation(frames))
    lnx = lncall(ipa_block.ln1,x, cond)
    x = x + ipa_block.ipa(T, lnx, T, lnx, zij = pair_feats, mask = mask, kwargs...) ./ 2
    x = x + ipa_block.ff(lncall(ipa_block.ln2,x, cond)) ./ 2
    return x
end

#MessagePassingIPA.jl:
function (ipa_block::IPAblock)(g, frames::Rigid, x, pair_feats; cond = nothing)
    x = x + ipa_block.ipa(g, lncall(ipa_block.ln1,x, cond), pair_feats, frames) ./ 2
    x = x + ipa_block.ff(lncall(ipa_block.ln2,x, cond)) ./ 2
    return x
end


"""
   CrossFrameIPA(dim::Int, ipa; ln = Flux.LayerNorm(dim))

Constructs a layer that takes one embedding, and two sets of frames. Runs layernorm on the embedding, and then makes a cross-attention IPA call with
one embedding but two frames. Useful for self-conditioning where two sets of frames need to communicate with each other.
"""
struct CrossFrameIPA{A,B}
    ln::A
    ipa::B
end
Flux.@layer CrossFrameIPA
CrossFrameIPA(dim::Int, ipa; ln = Flux.LayerNorm(dim)) = CrossFrameIPA(ln, ipa)
function (ipa_block::CrossFrameIPA)(frames1::Rigid, frames2::Rigid, x; pair_feats = nothing, cond = nothing, mask = 0, kwargs...)
    T1 = values(linear(frames1)), values(translation(frames1))
    T2 = values(linear(frames2)), values(translation(frames2))
    lnx = Onion.lncall(ipa_block.ln, x, cond)
    x = x + ipa_block.ipa(T1, lnx, T2, lnx, zij = pair_feats, mask = mask, show_warnings = false, kwargs...) ./ 2
    return x
end
