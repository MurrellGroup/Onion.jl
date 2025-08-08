using Pkg
Pkg.activate(".")
using Revise

#]add https://github.com/MurrellGroup/Onion.jl/tree/bm
#Pkg.add(["BSON","Flowfusion", "Flux", "RandomFeatureMaps", "ForwardBackward", "Optimisers", "Plots", "CannotWaitForTheseOptimisers", "LearningSchedules", "Serialization", "JLD2", "StatsBase", "Einops", "CUDA", "cuDNN"])

using BSON, Flowfusion, Flux, RandomFeatureMaps, ForwardBackward, Optimisers, Plots, CannotWaitForTheseOptimisers, Onion, LearningSchedules, Serialization, JLD2, StatsBase, Einops, LinearAlgebra, Random

basedir = ""

GPUnum = 0
ENV["CUDA_VISIBLE_DEVICES"] = GPUnum
using CUDA

ATOM_TYPES = String["C", "CA", "CB", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3", "CG", "CG1", "CG2", "CH2", "CZ", "CZ2", "CZ3", "H", "H1", "H2", "H3", "HA", "HA2", "HA3", "HB", "HB1", "HB2", "HB3", "HD1", "HD11", "HD12", "HD13", "HD2", "HD21", "HD22", "HD23", "HD3", "HE", "HE1", "HE2", "HE21", "HE22", "HE3", "HG", "HG1", "HG11", "HG12", "HG13", "HG2", "HG21", "HG22", "HG23", "HG3", "HH", "HH11", "HH12", "HH2", "HH21", "HH22", "HZ", "HZ1", "HZ2", "HZ3", "N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ", "O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "OXT", "SD", "SG"]
AA_TO_ATOMS = Dict("Q" => ["C", "CA", "CB", "CD", "CG", "N", "NE2", "O", "OE1"], "W" => ["C", "CA", "CB", "CD1", "CD2", "CE2", "CE3", "CG", "CH2", "CZ2", "CZ3", "N", "NE1", "O"], "T" => ["C", "CA", "CB", "CG2", "N", "O", "OG1"], "P" => ["C", "CA", "CB", "CD", "CG", "N", "O"], "C" => ["C", "CA", "CB", "N", "O", "SG"], "V" => ["C", "CA", "CB", "CG1", "CG2", "N", "O"], "L" => ["C", "CA", "CB", "CD1", "CD2", "CG", "N", "O"], "M" => ["C", "CA", "CB", "CE", "CG", "N", "O", "SD"], "N" => ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"], "H" => ["C", "CA", "CB", "CD2", "CE1", "CG", "N", "ND1", "NE2", "O"], "A" => ["C", "CA", "CB", "N", "O"], "X" => ["C", "CA", "CB", "N", "O"], "D" => ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"], "G" => ["C", "CA", "N", "O"], "E" => ["C", "CA", "CB", "CD", "CG", "N", "O", "OE1", "OE2"], "Y" => ["C", "CA", "CB", "CD1", "CD2", "CE1", "CE2", "CG", "CZ", "N", "O", "OH"], "I" => ["C", "CA", "CB", "CD1", "CG1", "CG2", "N", "O"], "S" => ["C", "CA", "CB", "N", "O", "OG"], "K" => ["C", "CA", "CB", "CD", "CE", "CG", "N", "NZ", "O"], "R" => ["C", "CA", "CB", "CD", "CG", "CZ", "N", "NE", "NH1", "NH2", "O"], "F" => ["C", "CA", "CB", "CD1", "CD2", "CE1", "CE2", "CG", "CZ", "N", "O"])

atomtype_dict = Dict(zip(ATOM_TYPES, 1:length(ATOM_TYPES)))
atomtype_to_int(at::String) = get(atomtype_dict, at, 1)

function apply_random_rigid(X::AbstractArray{T}, σ::T=one(T)) where T<:Number
    @assert size(X, 1) == 3
    Q, _ = qr(randn!(similar(X, 3, 3)))
    if det(Q) < 0
        Q[:,1] .*= -1
    end
    R = Q
    t = randn!(similar(X, 3)) * σ
    X′ = reshape(X, 3, :)
    Y′ = R * X′ .+ t
    Y = reshape(Y′, size(X))
    return Y
end

function batchrecs(recs; maxatoms = 256, longest_res = 50)
    atom_names = ones(Int, maxatoms, length(recs))
    atom_xyz = ones(Float32, 3, maxatoms, length(recs))
    atom_chainids = ones(Int, maxatoms, length(recs))
    anchors = ones(Int, maxatoms, length(recs))
    anchor_xyz = zeros(Float32, 3, maxatoms, length(recs))
    for i in 1:length(recs)
        reclen = length(recs[i].atom_name)
        pre_startpos = rand(1:(reclen-(maxatoms + longest_res)))
        startpos = findfirst(recs[i].atom_name[pre_startpos:end] .== "N") + pre_startpos - 1
        atom_names[:, i] = atomtype_to_int.(recs[i].atom_name[startpos:startpos + maxatoms - 1])
        current_anchor = 2
        for j in 1:maxatoms
            anchors[j, i] = current_anchor
            if atom_names[j, i] == 2
                current_anchor = j
            end
        end
        xyz = apply_random_rigid(recs[i].atom_xyz[:, startpos:startpos + maxatoms - 1])
        xyz_mu = mean(xyz, dims = 2)
        atom_xyz[:, :, i] = xyz .- xyz_mu
        anchor_xyz[:, :, i] = xyz[:, anchors[:, i]] .- xyz_mu
        rec_chainids = recs[i].chainids[recs[i].atom_res]
        atom_chainids[:, i] = rec_chainids[startpos:startpos + maxatoms - 1]
    end
    fixed = (atom_names .== 64) .| (atom_names .== 2) .| (atom_names .== 1)
    (; atoms = atom_names, xyz = atom_xyz, chainids = atom_chainids, fixed, anchors, anchor_xyz)
end

function atomplot(xyz, fixed)
    fixyz = xyz[:,fixed]
    xmax, xmin = maximum(fixyz[1,:])+4, minimum(fixyz[1,:])-4
    ymax, ymin = maximum(fixyz[2,:])+4, minimum(fixyz[2,:])-4
    zmax, zmin = maximum(fixyz[3,:])+4, minimum(fixyz[3,:])-4
    pl = scatter(xyz[1,fixed], xyz[2,fixed], xyz[3,fixed], color = :blue, legend = :none, msw = 0, ms = 2,
        xlims = (xmin, xmax), ylims = (ymin, ymax), zlims = (zmin, zmax), size = (1000, 1000))
    dists = sqrt.(abs.(Onion.pairwise_sqeuclidean(permutedims(xyz, (2,1)), xyz)))
    bonds = (dists .> 1.15f0) .& (dists .< 1.6f0)
    for i in 1:size(bonds, 1)
        for j in 1:size(bonds, 2)
            if bonds[i,j]
                plot!(xyz[1,[i,j]], xyz[2,[i,j]], xyz[3,[i,j]], color = :grey, linewidth = 1)
            end
        end
    end
    scatter!(xyz[1,.!fixed], xyz[2,.!fixed], xyz[3,.!fixed], color = :red, legend = :none, msw = 0, ms = 1)
    pl
end


function pair_features(anchors, chainids)
    a = rearrange(Onion.batched_pairs(==, anchors, anchors), (..) --> (1, ..))
    c = rearrange(Onion.batched_pairs(==, chainids, chainids), (..) --> (1, ..))
    return vcat(a, c)
end

struct SidechainerV1{L}
    layers::L
end
Flux.@layer SidechainerV1
function SidechainerV1(dim, depth, shifter_depth, heads)
    layers = (;
        dim, depth, shifter_depth,
        t_encoding = Chain(RandomFourierFeatures(1 => dim, 1f0), Dense(dim => dim)),
        atom_encoder = Embedding(83 => dim),
        transformers = [Onion.AdaSTRINGTransformerBlock(dim, dim, heads, 3) for _ in 1:depth],
         # transformers = [Onion.AdaTransformerBlock(dim, dim, heads) for _ in 1:depth],
        shifters = [Dense(dim => 3, bias = false) for _ in 1:shifter_depth],
        pair_heads = [Dense(3 => heads) for _ in 1:depth],
    )
    return SidechainerV1(layers)
end
function (m::SidechainerV1)(t, Xt, C)
    l = m.layers
    xyz = tensor(Xt)
    x = l.atom_encoder(C.atoms) #.+ l.pos_encoding(xyz)
    t_cond = l.t_encoding(rearrange(t, (..) --> (1, ..)))
    pf = vcat(rearrange(Onion.pairwise_sqeuclidean(permutedims(xyz, (2,1,3)), xyz), (..) --> (1, ..)), C.pf)
    for (i,tr) in enumerate(l.transformers)
        dpf = rearrange(l.pair_heads[i](pf), (:pd, :l1, :l2, :b) --> (:l1, :l2, (:pd, :b))) 
        x = tr(x, xyz, t_cond, mask = dpf)
        # rope = MultidimRoPE()
        # x = tr(x, xyz, t_cond; rope=rope, mask=dpf)
        if i > l.depth - l.shifter_depth
            xyz += l.shifters[i-(l.depth - l.shifter_depth)](x) .* (rearrange(Xt.cmask, (..) --> (1, ..))) .* (1.05f0 .- rearrange(t, (..) --> (1, 1, ..)))
            if i != l.depth
                pf = vcat(rearrange(Onion.pairwise_sqeuclidean(permutedims(xyz, (2,1,3)), xyz), (..) --> (1, ..)), C.pf)
            end
        end
    end
    return xyz
end

BSON.@load basedir*"pdb500_allatom.bson" allatom_dataset;
legal_inds = findall([length(d.atom_name) > 356 for d in allatom_dataset])

#b = batchrecs(allatom_dataset[sample(legal_inds, 1, replace = false)]; 128)
#X1 = MaskedState(ContinuousState(b.xyz), Array{Bool}(.!b.fixed), Array{Bool}(.!b.fixed))
#atomplot(tensor(X1)[:,:,1], b.fixed[:,1])

X0sample(b) = (.! reshape(b.fixed, 1, size(b.fixed)...)) .* (b.anchor_xyz .+ 0.3f0 .* randn(Float32, size(b.xyz)...)) .+ reshape(b.fixed, 1, size(b.fixed)...) .* b.xyz

P = BrownianMotion(0.2f0)
model = SidechainerV1(384, 10, 6, 8) |> gpu
for l in model.layers.shifters
    l.weight ./= 20
end

scheduler = burnin_learning_schedule(0.0001f0, 0.005f0, 1.1f0, 0.999f0)
opt_state = Flux.setup(Muon(eta = scheduler.lr), model)

legal_inds = findall([length(d.atom_name) > 612 for d in allatom_dataset])
batch_size = 16
maxatoms = 512

iters = 100000
losses = []
for i in 1:iters
    b = batchrecs(allatom_dataset[sample(legal_inds, batch_size, replace = false)]; maxatoms)
    X0 = ContinuousState(X0sample(b))
    X1 = MaskedState(ContinuousState(b.xyz), Array{Bool}(.!b.fixed), Array{Bool}(.!b.fixed))
    t = rand(Float32, batch_size)
    Xt = bridge(P, X0, X1, t)
    C = (; atoms = b.atoms, anchors = b.anchors, chainids = b.chainids, pf = Float32.(pair_features(b.anchors, b.chainids)))
    t, Xt, C, X1 = (t, Xt, C, X1) |> gpu
    l,g = Flux.withgradient(model) do m
        floss(P, m(t,Xt, C), X1, scalefloss(P, t, 2, 0.2f0))
    end
    Flux.update!(opt_state, model, g[1])
    if i % 10 == 0
        Flux.adjust!(opt_state, next_rate(scheduler))
    end
    println("i: $i; Loss: $l; lr: $(scheduler.lr)")
    push!(losses, l)
    if i % 10000 == 0
        let model = model |> cpu
            @save basedir*"naiveeq2_sidechainflow_checkpoint_batch$(i).jld2" model
        end
    end
    if i % 1000 == 0
        pl = plot(losses, xlabel = "Batch", ylabel = "Loss", legend = :none)
        savefig(pl, basedir*"naiveeq2_loss.pdf")
        gen_b = batchrecs(allatom_dataset[sample(legal_inds, 1, replace = false)]; maxatoms)
        C = (; atoms = gen_b.atoms, anchors = gen_b.anchors, chainids = gen_b.chainids, pf = Float32.(pair_features(gen_b.anchors, gen_b.chainids)))
        X0 = MaskedState(ContinuousState(X0sample(gen_b)), .!gen_b.fixed, .!gen_b.fixed)
        paths = Tracker()
        samp = gen(P, X0, (t, Xt) -> cpu(model(gpu([t]), gpu(Xt), gpu(C))), 0f0:0.01f0:1f0, tracker = paths)
        anim = @animate for i in 1:length(paths.t)
            atomplot(tensor(paths.xt[i][1])[:,:,1], .!paths.xt[i][1].cmask[:,1])
        end
        gif(anim, basedir*"naiveeq2_quicktrained_$(i).mp4", fps = 15)
    end
end

running_median = [median(losses[i:i+100]) for i in 1:length(losses)-100]
pl = plot(losses, xlabel = "Batch", ylabel = "Loss", legend = :none, ylims = (0,0.5))
plot!(running_median, color = :red, label = "Running Median")
savefig(pl, basedir*"naiveeq2_loss_thin.pdf")

#Silly, but we need to mod this to not throw atoms away - original one should have this as a flag instead.
function fullbatch(recs)
    maxatoms = length(recs[1].atom_name)
    atom_names = ones(Int, maxatoms, length(recs))
    atom_xyz = ones(Float32, 3, maxatoms, length(recs))
    atom_chainids = ones(Int, maxatoms, length(recs))
    anchors = ones(Int, maxatoms, length(recs))
    anchor_xyz = zeros(Float32, 3, maxatoms, length(recs))
    for i in 1:length(recs)
        atom_names[:, i] = atomtype_to_int.(recs[i].atom_name)
        @assert atom_names[2,1] == 2
        current_anchor = 2
        for j in 1:maxatoms
            anchors[j, i] = current_anchor
            if atom_names[j, i] == 2
                current_anchor = j
            end
        end
        xyz = apply_random_rigid(recs[i].atom_xyz)
        xyz_mu = mean(xyz, dims = 2)
        atom_xyz[:, :, i] = xyz .- xyz_mu
        anchor_xyz[:, :, i] = xyz[:, anchors[:, i]] .- xyz_mu
        rec_chainids = recs[i].chainids[recs[i].atom_res]
        atom_chainids[:, i] = rec_chainids
    end
    fixed = (atom_names .== 64) .| (atom_names .== 2) .| (atom_names .== 1)
    (; atoms = atom_names, xyz = atom_xyz, chainids = atom_chainids, fixed, anchors, anchor_xyz)
end

for ind = 1:length(allatom_dataset)
    if length(allatom_dataset[ind].atom_name) < 4000
        @show ind
        gen_b = fullbatch(allatom_dataset[[ind]])
        C = (; atoms = gen_b.atoms, anchors = gen_b.anchors, chainids = gen_b.chainids, pf = Float32.(pair_features(gen_b.anchors, gen_b.chainids)))
        X0 = MaskedState(ContinuousState(X0sample(gen_b)), .!gen_b.fixed, .!gen_b.fixed)
        paths = Tracker()
        samp = gen(P, X0, (t, Xt) -> cpu(model(gpu([t]), gpu(Xt), gpu(C))), 0f0:0.005f0:1f0, tracker = paths)
        anim = @animate for i in 1:length(paths.t)+10
            @show i
            if i <= length(paths.t)
                atomplot(tensor(paths.xt[i][1])[:,:,1], .!paths.xt[i][1].cmask[:,1])
            else
                atomplot(tensor(samp)[:,:,1], .!samp.cmask[:,1])
            end
        end
        gif(anim, basedir*"naiveeq2_gen_rec_Xt_$(ind).mp4", fps = 10)
        gif(anim, basedir*"naiveeq2_gen_rec_Xt_$(ind).gif", fps = 10)
        anim = @animate for i in 1:length(paths.t)+10
            @show i
            if i <= length(paths.t)
                atomplot(tensor(paths.x̂1[i][1])[:,:,1], .!paths.xt[i][1].cmask[:,1])
            else
                atomplot(tensor(samp)[:,:,1], .!samp.cmask[:,1])
            end
        end
        gif(anim, basedir*"naiveeq2_gen_rec_X1hat_$(ind).mp4", fps = 10)
        gif(anim, basedir*"naiveeq2_gen_rec_X1hat_$(ind).gif", fps = 10)
        anim = @animate for i in 1:30
            @show i
            if (i < 5) || (i > 25)
                X1 = MaskedState(ContinuousState(gen_b.xyz), .!gen_b.fixed, .!gen_b.fixed)
                atomplot(tensor(X1)[:,:,1], gen_b.fixed[:,1])
            else
                X0 = MaskedState(ContinuousState(X0sample(gen_b)), .!gen_b.fixed, .!gen_b.fixed)
                samp = gen(P, X0, (t, Xt) -> cpu(model(gpu([t]), gpu(Xt), gpu(C))), 0f0:0.005f0:1f0)
                atomplot(tensor(samp)[:,:,1], .!samp.cmask[:,1])
            end
        end
        gif(anim, basedir*"naiveeq2_multigens_$(ind).mp4", fps = 2)
        gif(anim, basedir*"naiveeq2_multigens_$(ind).gif", fps = 2)
    end
end
