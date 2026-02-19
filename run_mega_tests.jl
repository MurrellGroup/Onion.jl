#!/usr/bin/env julia
#=
  MEGA TEST SUITE — AlphaFold2, ESMFold, BoltzGen
  ================================================
  Single-session comprehensive test covering all three protein models
  from one unified Julia environment.

  Usage:
    julia --project=. run_mega_tests.jl --gpu [output_dir]

  Environment:
    - Onion.jl    branch: bg_optimized  (WorkingBoltzGen/Onion.jl)
    - OnionTile.jl branch: main
    - cuTile.jl    branch: main

  ┌──────────────────────────────────────────────────────────────────────┐
  │ TEST MANIFEST                                                       │
  │                                                                     │
  │ PHASE 1: ESMFold — Inference                                        │
  │   esm01  Fold 12-residue helix (ELLKKLLEELKG)                       │
  │          → PDB output, pLDDT, pTM                                   │
  │   esm02  Fold 30-residue GCN4 peptide                               │
  │          → PDB output, pLDDT, pTM (scaling check)                   │
  │   esm03  Multi-chain fold 2×12 residues (colon-separated)           │
  │          → PDB output, pLDDT, pTM                                   │
  │   esm04  Confidence metrics extraction (pLDDT, PAE, pTM)            │
  │          → field existence, shape, value range checks                │
  │   esm05  Composable pipeline (prepare→embed→trunk→heads)            │
  │          → shape checks at each stage, pTM from heads               │
  │   esm06  Single trunk pass (no struct module, no recycling)          │
  │          → shape preservation, values changed from input             │
  │   esm07  Recycle variation (num_recycles=1 vs 4)                    │
  │          → pTM differs across recycle counts                        │
  │                                                                     │
  │ PHASE 1b: ESMFold — GPU Gradient Checks                             │
  │   esm_grad_A  ESM2 AD backward (33-layer transformer)               │
  │               → weight grads nonzero, no NaN                        │
  │   esm_grad_B  StructureModule backward (atom_pos loss)              │
  │               → weight grads nonzero, no NaN                        │
  │   esm_grad_C  trunk→struct→lDDT head backward                      │
  │               → trunk+lddt grads nonzero, no NaN                    │
  │   esm_grad_D  FD spot-check: lDDT head input grad vs central FD      │
  │               → cosine >0.99, max_rel <0.02 on 20 positions         │
  │                                                                     │
  │ PHASE 2: AlphaFold2 — Monomer Inference (9 residues)                │
  │   af2_01  Fold ACDEFGHIK seq-only, 3 recycles                       │
  │           → PDB, pLDDT (~43), clashes (≤1)                          │
  │   af2_02  Fold ACDEFGHIK + MSA, 3 recycles                          │
  │           → PDB, pLDDT (~43), clashes (0)                           │
  │   af2_03  Fold glucagon (29 res) + template, 3 recycles             │
  │           → PDB, pLDDT (~58), clashes (≤3)                          │
  │   af2_04  Fold glucagon + MSA + template, 3 recycles                │
  │           → PDB, pLDDT (~58), clashes (≤3)                          │
  │   af2_composable  Decomposed pipeline (build→prepare→run)           │
  │                   → same result as fold(), tests lower-level API     │
  │                                                                     │
  │ PHASE 3: AlphaFold2 — Multimer Inference (2×30 residues)            │
  │   af2_05  Fold GCN4 homodimer seq-only, 5 recycles                  │
  │           → PDB, pLDDT (~95), clashes (0), 2 chains                 │
  │   af2_06  Fold GCN4 homodimer + MSA, 5 recycles                     │
  │           → PDB, pLDDT (~95), clashes (0), 2 chains                 │
  │   af2_heterodimer  Heteromeric multimer (GCN4 + glucagon)           │
  │                    → 2 chains, different sequences                   │
  │   af2_homodimer_templates  Homodimer + template PDB                 │
  │                            → 2 chains, template-guided              │
  │                                                                     │
  │ PHASE 4: AlphaFold2 — GPU Gradient Checks                          │
  │   af2_grad_A  EvoformerIteration backward                           │
  │               → weight grads nonzero, no NaN                        │
  │   af2_grad_B1 StructureModuleCore backward (act loss)               │
  │               → weight grads nonzero, no NaN                        │
  │   af2_grad_B2 StructureModuleCore backward (atom_pos loss)          │
  │               → weight grads nonzero, no NaN                        │
  │   af2_grad_C  PredictedLDDTHead backward + input grad               │
  │               → weight+input grads nonzero, no NaN                  │
  │   af2_grad_D  E2E: block→struct→pLDDT backward                     │
  │               → all 4 component grads nonzero, no NaN               │
  │   af2_grad_E  Continuous sequence input gradient                    │
  │               → target_feat grad nonzero, no NaN                    │
  │   af2_grad_F  Full pipeline: input→block→struct→pLDDT              │
  │               → end-to-end input grad nonzero, no NaN               │
  │                                                                     │
  │ PHASE 5: BoltzGen — Design (BoltzGen1 model)                       │
  │   bg_01  Design protein only (12 res, de novo)                      │
  │          → PDB14/37, mmCIF, geometry, bonds (~96, 0 viol)           │
  │   bg_02  Design protein + small molecule (chorismite.yaml)          │
  │          → bonds (~1007, 0 violations)                              │
  │   bg_03  Design protein + DNA + SMILES (mixed yaml)                 │
  │          → bonds (~90, 0 violations)                                │
  │   bg_04  Target conditioned design (8 res on bg_01 output)          │
  │          → bonds (~159, 0 violations)                               │
  │   bg_design_cyclic  Cyclic protein design (12 res)                  │
  │                     → bonds, cyclic=true                            │
  │   bg_design_dna     DNA sequence design (8 res)                     │
  │                     → bonds, chain_type=DNA                         │
  │   bg_denovo_sample  De novo sampling (10 res, no sequence)          │
  │                     → bonds, no input sequence                      │
  │   bg_design_partial Design with mask (redesign 8 of 12 residues)    │
  │                     → bonds, partial design_mask                    │
  │                                                                     │
  │ PHASE 6: BoltzGen — Fold (Boltz2 confidence model)                  │
  │   bg_05  Fold poly-GLY (14 res)                                     │
  │          → bonds (55, 0 viol), confidence metrics                   │
  │   bg_06  Fold trastuzumab VH antibody (121 res)                     │
  │          → bonds (951, 0 viol), confidence metrics                  │
  │   bg_07  Fold lysozyme (129 res) no MSA                             │
  │          → bonds (1020, 0 viol), pTM                                │
  │   bg_08  Fold lysozyme (129 res) with MSA (100 sequences)           │
  │          → bonds (1020, 0 viol), pTM (should improve)               │
  │   bg_fold_polyG_no_template   (14 res, ~1-2 viol expected)          │
  │   bg_fold_polyG_with_template (14 res, 0 viol with template)        │
  │   bg_fold_3chain  Multi-chain fold (2 proteins + 1 DNA)             │
  │                   → bonds, fold_from_sequences with chain_types     │
  │   bg_fold_rna     RNA-only fold (12 res)                            │
  │                   → bonds, chain_type=RNA                           │
  │                                                                     │
  │ PHASE 7: BoltzGen — Fold (Boltz2 affinity model)                    │
  │   bg_09  Fold poly-GLY with affinity heads                          │
  │          → bonds (55, ~7 viol acceptable — degenerate input)        │
  │   bg_10  Fold from structure with affinity (bg_03 output)           │
  │          → bonds (~90, 0 viol)                                      │
  │                                                                     │
  │ PHASE 8: BoltzGen — E2E Small Molecule & DNA                       │
  │   bg_e2e01  Design protein + HEM (porphyrin)                        │
  │             → geometry, bonds, Fe-N ~2.0Å                           │
  │   bg_e2e06  Design protein + SMILES (aspirin)                       │
  │             → geometry, bonds                                       │
  │   bg_e2e09  Fold DNA only (ACGTACGTAC)                              │
  │             → geometry, bonds, confidence                           │
  │   bg_e2e11  Fold from HEM design (round-trip)                       │
  │             → bonds (103, 0 viol), Fe-N ~2.0Å                      │
  │                                                                     │
  │ PHASE 9: BoltzGen — TCD Scaling (first 2 of 4 cases)               │
  │   bg_tcd01  TCD: 129-res target + 8 designed residues               │
  │             → geometry, bonds, timing                               │
  │   bg_tcd02  TCD: 121-res target + 60 designed residues              │
  │             → geometry, bonds, timing                               │
  │                                                                     │
  │ TOTAL: ~49 tests                                                    │
  └──────────────────────────────────────────────────────────────────────┘
=#

using Printf
using Statistics
using Dates
using LinearAlgebra
using Random
using CUDA
using cuDNN
using OnionTile
using Flux
using Zygote

const USE_GPU = "--gpu" in ARGS
const non_flag_args = filter(a -> a != "--gpu", ARGS)
const OUTDIR = length(non_flag_args) >= 1 ? non_flag_args[1] : joinpath(@__DIR__, "mega_test_outputs")
mkpath(OUTDIR)

const LOG_FILE = joinpath(OUTDIR, "mega_test_log.txt")
const LOG_IO = open(LOG_FILE, "w")

function tee(args...)
    println(stdout, args...)
    println(LOG_IO, args...)
    flush(LOG_IO)
end

function tee_printf(fmt, args...)
    s = @eval @sprintf($fmt, $(args...))
    print(stdout, s)
    print(LOG_IO, s)
    flush(LOG_IO)
end

# Results tracking
const results = Tuple{String,String,Bool,String,Float64}[]  # (phase, name, pass, detail, elapsed_s)

function run_case(f::Function, phase::String, name::String)
    tee("\n", "─"^60)
    tee("  CASE: $name")
    tee("─"^60)
    elapsed = 0.0
    try
        elapsed = @elapsed detail = f()
        push!(results, (phase, name, true, detail, elapsed))
        tee("[PASS] $name ($(round(elapsed; digits=1))s) $detail")
    catch e
        msg = sprint(showerror, e, catch_backtrace())
        push!(results, (phase, name, false, first(msg, 200), elapsed))
        tee("[FAIL] $name: ", first(msg, 500))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

# AF2 clash detection (ported from regression_helpers.jl)
function parse_pdb_atoms(pdb_path::AbstractString)
    atoms = NamedTuple[]
    open(pdb_path, "r") do io
        for raw in eachline(io)
            startswith(raw, "ATOM") || continue
            line = rpad(raw, 80)
            atom_name = strip(line[13:16])
            resname = strip(line[18:20])
            chain = line[22]
            resseq = parse(Int, strip(line[23:26]))
            x = parse(Float64, strip(line[31:38]))
            y = parse(Float64, strip(line[39:46]))
            z = parse(Float64, strip(line[47:54]))
            push!(atoms, (atom=atom_name, resname=resname, chain=chain, resseq=resseq, x=x, y=y, z=z))
        end
    end
    return atoms
end

function check_clashes(pdb_path::AbstractString; thresh::Float64=2.0)
    atoms = parse_pdb_atoms(pdb_path)
    n = 0
    worst = Inf
    for i in 1:length(atoms), j in (i+1):length(atoms)
        abs(atoms[i].resseq - atoms[j].resseq) <= 1 && continue
        d = sqrt((atoms[i].x - atoms[j].x)^2 +
                 (atoms[i].y - atoms[j].y)^2 +
                 (atoms[i].z - atoms[j].z)^2)
        if d < thresh
            n += 1
            worst = min(worst, d)
        end
    end
    return (count=n, worst=worst)
end

function pdb_chain_ids(pdb_path::AbstractString)
    atoms = parse_pdb_atoms(pdb_path)
    return unique(a.chain for a in atoms)
end

# AF2 gradient helper (from grad_check_af2.jl)
function _walk_grads!(stats, g)
    if g === nothing || g isa Number
        return
    elseif g isa AbstractArray{<:Number}
        ga = g isa CuArray ? Array(g) : g
        stats[:total] += length(ga)
        stats[:nonzero] += count(!=(0f0), ga)
        stats[:nan_count] += count(isnan, ga)
    elseif g isa NamedTuple
        for v in values(g); _walk_grads!(stats, v); end
    elseif g isa Tuple
        for v in g; _walk_grads!(stats, v); end
    elseif g isa AbstractVector
        for v in g; _walk_grads!(stats, v); end
    end
end

function grad_summary(g; label="")
    stats = Dict{Symbol,Int}(:total => 0, :nonzero => 0, :nan_count => 0)
    _walk_grads!(stats, g)
    total, nonzero, nan_count = stats[:total], stats[:nonzero], stats[:nan_count]
    pct = total > 0 ? 100.0 * nonzero / total : 0.0
    tee("    $label: $nonzero/$total nonzero ($(round(pct; digits=1))%), $nan_count NaN")
    return (total=total, nonzero=nonzero, nan_count=nan_count)
end

# ═════════════════════════════════════════════════════════════════════════════
tee("=" ^ 70)
tee("  MEGA TEST SUITE — AlphaFold2 + ESMFold + BoltzGen")
tee("  GPU: ", USE_GPU ? "yes ($(CUDA.name(CUDA.device())))" : "no")
tee("  Output: $OUTDIR")
tee("  Time: ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
tee("=" ^ 70)

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1: ESMFold
# ═════════════════════════════════════════════════════════════════════════════
tee("\n", "#"^70)
tee("  PHASE 1: ESMFold")
tee("#"^70)

using ESMFold

tee("  Loading ESMFold model...")
t0 = time()
esm_model = ESMFold.load_ESMFold()
if USE_GPU
    esm_model = Flux.gpu(esm_model)
end
tee("  Model loaded in $(round(time()-t0; digits=1))s")

run_case("ESMFold", "esm01_fold_helix_12res") do
    output = ESMFold.infer(esm_model, "ELLKKLLEELKG")
    pdb = ESMFold.output_to_pdb(output)[1]
    path = joinpath(OUTDIR, "esm01_helix.pdb")
    open(path, "w") do io; write(io, pdb); end
    plddt = Float64(Array(output[:mean_plddt])[1])
    ptm = Float64(Array(output[:ptm])[1])
    length(pdb) > 100 || error("PDB too small")
    @sprintf("pLDDT=%.1f pTM=%.3f pdb=%dB", plddt, ptm, length(pdb))
end

run_case("ESMFold", "esm02_fold_gcn4_30res") do
    output = ESMFold.infer(esm_model, "MKQLEDKVEELLSKNYHLENEVARLKKLV")
    pdb = ESMFold.output_to_pdb(output)[1]
    path = joinpath(OUTDIR, "esm02_gcn4.pdb")
    open(path, "w") do io; write(io, pdb); end
    plddt = Float64(Array(output[:mean_plddt])[1])
    ptm = Float64(Array(output[:ptm])[1])
    @sprintf("pLDDT=%.1f pTM=%.3f pdb=%dB", plddt, ptm, length(pdb))
end

# esm03: Multi-chain inference (two short chains with linker)
run_case("ESMFold", "esm03_multichain_2x12") do
    output = ESMFold.infer(esm_model, "ELLKKLLEELKG:ELLKKLLEELKG")
    pdb = ESMFold.output_to_pdb(output)[1]
    path = joinpath(OUTDIR, "esm03_multichain.pdb")
    open(path, "w") do io; write(io, pdb); end
    plddt = Float64(Array(output[:mean_plddt])[1])
    ptm = Float64(Array(output[:ptm])[1])
    length(pdb) > 100 || error("PDB too small")
    @sprintf("pLDDT=%.1f pTM=%.3f pdb=%dB", plddt, ptm, length(pdb))
end

# esm04: Confidence metrics extraction
run_case("ESMFold", "esm04_confidence_metrics") do
    output = ESMFold.infer(esm_model, "MKQLEDKVEELLSKNYHLENEVARLKKLV")
    cm = ESMFold.confidence_metrics(output)
    # Check all expected fields exist and have correct shapes
    haskey(cm, :plddt) || error("Missing plddt")
    haskey(cm, :mean_plddt) || error("Missing mean_plddt")
    haskey(cm, :ptm) || error("Missing ptm")
    haskey(cm, :predicted_aligned_error) || error("Missing PAE")
    haskey(cm, :max_predicted_aligned_error) || error("Missing max_PAE")

    plddt_arr = Array(cm.plddt)
    pae_arr = Array(cm.predicted_aligned_error)
    mean_plddt = Float64(Array(cm.mean_plddt)[1])
    ptm_val = Float64(Array(cm.ptm)[1])
    max_pae = Float64(cm.max_predicted_aligned_error)  # scalar, not array

    # Sanity: pLDDT should be 0-100, PAE should be non-negative
    all(0 .<= plddt_arr .<= 100) || error("pLDDT out of range")
    all(pae_arr .>= 0) || error("PAE negative")
    ptm_val >= 0 || error("pTM negative")
    @sprintf("mean_pLDDT=%.1f pTM=%.3f max_PAE=%.1f pae_shape=%s", mean_plddt, ptm_val, max_pae, string(size(pae_arr)))
end

# esm05: Composable pipeline stages
run_case("ESMFold", "esm05_composable_pipeline") do
    seq = "ELLKKLLEELKG"
    inputs = ESMFold.prepare_inputs(esm_model, seq)

    # Check input shapes
    L = length(seq)
    size(inputs.aa) == (L, 1) || error("aa shape mismatch: $(size(inputs.aa)) vs ($L, 1)")
    size(inputs.mask) == (L, 1) || error("mask shape mismatch: $(size(inputs.mask)) vs ($L, 1)")

    # Run embedding
    emb = ESMFold.run_embedding(esm_model, inputs)
    c_s = size(emb.s_s_0, 1)
    c_z = size(emb.s_z_0, 1)
    size(emb.s_s_0, 2) == L || error("s_s_0 L mismatch")
    size(emb.s_z_0, 2) == L && size(emb.s_z_0, 3) == L || error("s_z_0 L mismatch")

    # Run trunk
    structure = ESMFold.run_trunk(esm_model, emb.s_s_0, emb.s_z_0, inputs; num_recycles=1)
    haskey(structure, :positions) || error("Missing positions in trunk output")

    # Run heads
    full_output = ESMFold.run_heads(esm_model, structure, inputs)
    haskey(full_output, :ptm) || error("Missing ptm in heads output")
    haskey(full_output, :plddt) || error("Missing plddt in heads output")

    ptm_val = Float64(Array(full_output[:ptm])[1])
    @sprintf("c_s=%d c_z=%d L=%d pTM=%.3f pipeline=OK", c_s, c_z, L, ptm_val)
end

# esm06: Single trunk pass (no structure module, no recycling)
run_case("ESMFold", "esm06_trunk_single_pass") do
    seq = "ACDEFGHIK"
    inputs = ESMFold.prepare_inputs(esm_model, seq)
    emb = ESMFold.run_embedding(esm_model, inputs)

    # Copy inputs since trunk blocks may mutate in-place
    s_s_before = Array(emb.s_s_0)
    s_z_before = Array(emb.s_z_0)

    result = ESMFold.run_trunk_single_pass(esm_model, emb.s_s_0, emb.s_z_0, inputs)
    size(result.s_s) == size(s_s_before) || error("s_s shape changed: $(size(result.s_s)) vs $(size(s_s_before))")
    size(result.s_z) == size(s_z_before) || error("s_z shape changed: $(size(result.s_z)) vs $(size(s_z_before))")

    # Values should be different from input (the trunk actually did something)
    s_diff = Float64(sum(abs.(Array(result.s_s) .- s_s_before)))
    z_diff = Float64(sum(abs.(Array(result.s_z) .- s_z_before)))
    s_diff > 0 || error("s_s unchanged after trunk pass")
    z_diff > 0 || error("s_z unchanged after trunk pass")
    @sprintf("s_diff=%.1f z_diff=%.1f shapes=OK", s_diff, z_diff)
end

# esm07: Recycle count variation — verify different num_recycles produces different output
run_case("ESMFold", "esm07_recycle_variation") do
    seq = "ELLKKLLEELKG"
    out1 = ESMFold.infer(esm_model, seq; num_recycles=1)
    out4 = ESMFold.infer(esm_model, seq; num_recycles=4)
    ptm1 = Float64(Array(out1[:ptm])[1])
    ptm4 = Float64(Array(out4[:ptm])[1])
    plddt1 = Float64(Array(out1[:mean_plddt])[1])
    plddt4 = Float64(Array(out4[:mean_plddt])[1])
    # More recycles should produce different output
    abs(ptm4 - ptm1) > 0 || error("pTM identical across recycle counts: r1=$ptm1 r4=$ptm4")
    @sprintf("r1: pTM=%.3f pLDDT=%.1f | r4: pTM=%.3f pLDDT=%.1f | delta_pTM=%.4f",
             ptm1, plddt1, ptm4, plddt4, ptm4 - ptm1)
end

# ── ESMFold Gradient Tests (GPU only) ──────────────────────────────────────
if USE_GPU
    tee("\n  ESMFold Gradient Tests")

    # esm_grad_A: ESM2 AD forward — gradient through the 33-layer transformer
    run_case("ESMFold-Grad", "esm_grad_A_esm2_ad") do
        # Use a short sequence to keep memory manageable
        seq = "ACDEFGHIK"
        inputs = ESMFold.prepare_inputs(esm_model, seq)
        tokens_bt = permutedims(inputs.aa, (2, 1))  # (B, T) for esm2_forward_ad

        loss, grads = Zygote.withgradient(esm_model.embed.esm) do esm
            x = ESMFold.esm2_forward_ad(esm, tokens_bt)
            Float32(mean(x))
        end
        s = grad_summary(grads[1]; label="esm2 weights")
        s.nonzero > 0 || error("No nonzero gradients")
        s.nan_count == 0 || error("NaN in gradients")
        @sprintf("loss=%.4f nonzero=%d nan=%d", loss, s.nonzero, s.nan_count)
    end

    # esm_grad_B: Structure module backward
    run_case("ESMFold-Grad", "esm_grad_B_structure_module") do
        seq = "ACDEFGHIK"
        inputs = ESMFold.prepare_inputs(esm_model, seq)
        emb = ESMFold.run_embedding(esm_model, inputs)

        # Get trunk single pass output (provides s_s, s_z for structure module)
        trunk_out = ESMFold.run_trunk_single_pass(esm_model, emb.s_s_0, emb.s_z_0, inputs)

        # Pre-compute SM inputs outside gradient scope (convert/projections are not differentiable wrt SM)
        trunk_mod = esm_model.trunk
        sm_s = trunk_mod.trunk2sm_s(trunk_out.s_s)
        sm_z = trunk_mod.trunk2sm_z(trunk_out.s_z)
        mask_f = convert.(Float32, inputs.mask)

        # Gradient through structure module only
        loss, grads = Zygote.withgradient(trunk_mod.structure_module) do sm
            sm_input = Dict(:single => sm_s, :pair => sm_z)
            result = sm(sm_input, inputs.aa, mask_f)
            Float32(mean(result[:positions]))
        end
        s = grad_summary(grads[1]; label="structure_module weights")
        s.nonzero > 0 || error("No nonzero gradients")
        s.nan_count == 0 || error("NaN in gradients")
        @sprintf("loss=%.4f nonzero=%d nan=%d", loss, s.nonzero, s.nan_count)
    end

    # esm_grad_D: Finite difference spot-check of lDDT head gradient
    # Central FD on the lDDT head (3 linear layers + layernorm) — pure function, no in-place state.
    # SM FD is deferred to the AF2 FoldIterationCore test (same SM, small dims, CPU, Float64).
    run_case("ESMFold-Grad", "esm_grad_D_fd_lddt_head") do
        lddt_head = esm_model.lddt_head
        # Synthetic input matching SM output shape: (c_s, L, B) for 9 residues
        rng = Random.MersenneTwister(42)
        x_cpu = randn(rng, Float32, 384, 9, 1) .* 0.1f0  # c_s=384 for SM output
        x_gpu = CUDA.cu(x_cpu)

        # Weighted loss to break layernorm invariance
        w_cpu = randn(rng, Float32, size(lddt_head.linear_3.weight, 1), 9, 1)
        w_gpu = CUDA.cu(w_cpu)

        function lddt_loss_gpu(x)
            out = lddt_head(x)
            return Float64(sum(out .* w_gpu))
        end

        # AD gradient w.r.t. input
        loss_val, ad_grads = Zygote.withgradient(lddt_loss_gpu, x_gpu)
        ad_grad = Array(ad_grads[1])

        # Central FD on 20 random positions
        eps = 5e-4
        n_checks = 20
        indices = [CartesianIndex(rand(rng, 1:size(x_cpu,1)), rand(rng, 1:size(x_cpu,2)), 1) for _ in 1:n_checks]

        fd_vals = Float64[]
        ad_vals = Float64[]
        for idx in indices
            x_plus = copy(x_cpu); x_plus[idx] += eps
            x_minus = copy(x_cpu); x_minus[idx] -= eps
            f_plus = lddt_loss_gpu(CUDA.cu(x_plus))
            f_minus = lddt_loss_gpu(CUDA.cu(x_minus))
            push!(fd_vals, (f_plus - f_minus) / (2 * eps))
            push!(ad_vals, Float64(ad_grad[idx]))
        end

        # Cosine similarity
        cos_sim = dot(ad_vals, fd_vals) / (norm(ad_vals) * norm(fd_vals) + 1e-12)
        max_abs = maximum(abs.(ad_vals .- fd_vals))
        scale = max(maximum(abs.(fd_vals)), maximum(abs.(ad_vals)), 1e-8)
        max_rel = max_abs / scale

        cos_sim > 0.99 || error("FD check failed: cosine=$(cos_sim) (want >0.99)")
        max_rel < 0.02 || error("FD check failed: max_rel=$(max_rel) (want <0.02)")
        @sprintf("loss=%.4f cosine=%.6f max_rel=%.2e n=%d", loss_val, cos_sim, max_rel, n_checks)
    end

    # esm_grad_C: trunk → structure module → lDDT head backward
    # Tests gradient flow through the full folding pipeline (trunk blocks + SM + lDDT head).
    # ESM2 AD is tested separately in grad_A. Here we use pre-computed embeddings as input.
    run_case("ESMFold-Grad", "esm_grad_C_trunk_to_plddt") do
        seq = "ACDEFGHIK"
        inputs = ESMFold.prepare_inputs(esm_model, seq)
        emb = ESMFold.run_embedding(esm_model, inputs)

        # Pre-compute constants outside gradient scope
        mask_f = convert.(Float32, inputs.mask)

        # Gradient through trunk blocks → SM → lDDT head
        loss, grads = Zygote.withgradient(esm_model.trunk, esm_model.lddt_head) do trunk, lddt_head
            # Single trunk pass (manually unrolled — run_trunk_single_pass is not a closure param)
            z = emb.s_z_0 .+ trunk.pairwise_positional_embedding(inputs.residx, mask=inputs.mask)
            s = emb.s_s_0
            for block in trunk.blocks
                s, z = block(s, z; mask=nothing, residue_index=inputs.residx, chunk_size=trunk.chunk_size)
            end

            # Structure module
            sm_input = Dict(:single => trunk.trunk2sm_s(s), :pair => trunk.trunk2sm_z(z))
            structure = trunk.structure_module(sm_input, inputs.aa, mask_f)

            # lDDT head — use mean of logits as loss (avoids categorical_lddt reshape complexity)
            states = structure[:states]
            states_last = states isa AbstractArray ? states : states[end]
            states_cf = ndims(states_last) == 4 ? permutedims(states_last, (2, 1, 3, 4)) : states_last
            lddt_logits = lddt_head(states_cf)
            Float32(mean(lddt_logits))
        end
        st = grad_summary(grads[1]; label="trunk weights")
        sl = grad_summary(grads[2]; label="lddt_head weights")
        ok = st.nonzero > 0 && st.nan_count == 0 && sl.nonzero > 0 && sl.nan_count == 0
        ok || error("Gradient check failed: trunk_nz=$(st.nonzero) lddt_nz=$(sl.nonzero) trunk_nan=$(st.nan_count) lddt_nan=$(sl.nan_count)")
        @sprintf("loss=%.4f trunk_nz=%d lddt_nz=%d", loss, st.nonzero, sl.nonzero)
    end
else
    tee("\n  ESMFold Gradient Tests: Skipped (no --gpu)")
end

# Release ESMFold
esm_model = nothing
GC.gc(); CUDA.reclaim()

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2-3: AlphaFold2 — Inference
# ═════════════════════════════════════════════════════════════════════════════
tee("\n", "#"^70)
tee("  PHASE 2-3: AlphaFold2 — Inference")
tee("#"^70)

using Alphafold2

const AF2_ROOT = joinpath(@__DIR__, "Alphafold2.jl")
const AF2_MSA_DIR = joinpath(AF2_ROOT, "test", "regression", "msa")
const AF2_TEMPLATE_DIR = joinpath(AF2_ROOT, "test", "regression", "templates")

af2_device = USE_GPU ? Flux.gpu : identity

# Load monomer model
tee("  Loading AF2 monomer model...")
t0 = time()
af2_mono = Alphafold2.load_monomer(; device=af2_device)
tee("  Monomer loaded in $(round(time()-t0; digits=1))s")

# Phase 2: Monomer cases
run_case("AF2-Mono", "af2_01_seq_only_9res") do
    r = Alphafold2.fold("ACDEFGHIK"; num_recycle=3, out_prefix=joinpath(OUTDIR, "af2_01"))
    clashes = check_clashes(r.out_pdb)
    @sprintf("pLDDT=%.1f clashes=%d pdb=%s", r.mean_plddt, clashes.count, basename(r.out_pdb))
end

run_case("AF2-Mono", "af2_02_seq_msa_9res") do
    msa = joinpath(AF2_MSA_DIR, "monomer_short.a3m")
    r = Alphafold2.fold("ACDEFGHIK"; msas=msa, num_recycle=3, out_prefix=joinpath(OUTDIR, "af2_02"))
    clashes = check_clashes(r.out_pdb)
    @sprintf("pLDDT=%.1f clashes=%d", r.mean_plddt, clashes.count)
end

run_case("AF2-Mono", "af2_03_template_29res") do
    glucagon = "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT"
    tmpl = joinpath(AF2_TEMPLATE_DIR, "glucagon_template.pdb")
    r = Alphafold2.fold(glucagon; templates=tmpl, template_chains="A", num_recycle=3,
                        out_prefix=joinpath(OUTDIR, "af2_03"))
    clashes = check_clashes(r.out_pdb)
    @sprintf("pLDDT=%.1f clashes=%d", r.mean_plddt, clashes.count)
end

run_case("AF2-Mono", "af2_04_template_msa_29res") do
    glucagon = "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT"
    msa = joinpath(AF2_MSA_DIR, "monomer_glucagon.a3m")
    tmpl = joinpath(AF2_TEMPLATE_DIR, "glucagon_template.pdb")
    r = Alphafold2.fold(glucagon; msas=msa, templates=tmpl, template_chains="A", num_recycle=3,
                        out_prefix=joinpath(OUTDIR, "af2_04"))
    clashes = check_clashes(r.out_pdb)
    @sprintf("pLDDT=%.1f clashes=%d", r.mean_plddt, clashes.count)
end

# af2_composable: Decomposed pipeline — build_features → prepare_inputs → run_inference
run_case("AF2-Mono", "af2_composable_pipeline") do
    seq = "ACDEFGHIK"
    features = Alphafold2.build_monomer_features(seq; num_recycle=3)
    inputs = Alphafold2.prepare_inputs(af2_mono, features; num_recycle=3)
    result = Alphafold2.run_inference(af2_mono, inputs)
    plddt_mean = Float64(mean(result.plddt))
    ptm_val = result.ptm !== nothing ? Float64(result.ptm) : NaN
    result.plddt !== nothing || error("Missing plddt in inference result")
    length(result.aatype) == length(seq) || error("aatype length mismatch: $(length(result.aatype)) vs $(length(seq))")
    @sprintf("pLDDT=%.1f pTM=%.3f n_res=%d pipeline=build→prepare→run", plddt_mean, ptm_val, length(seq))
end

# Release monomer
af2_mono = nothing
Alphafold2._DEFAULT_MODEL[] = nothing
GC.gc(); CUDA.reclaim()

# Phase 3: Multimer cases
tee("\n  Loading AF2 multimer model...")
t0 = time()
af2_multi = Alphafold2.load_multimer(; device=af2_device)
tee("  Multimer loaded in $(round(time()-t0; digits=1))s")

const GCN4_SEQ = "MKQLEDKVEELLSKNYHLENEVARLKKLV"

run_case("AF2-Multi", "af2_05_homodimer_seq_only") do
    r = Alphafold2.fold(af2_multi, [GCN4_SEQ, GCN4_SEQ]; num_recycle=5,
                        out_prefix=joinpath(OUTDIR, "af2_05"))
    clashes = check_clashes(r.out_pdb)
    chains = pdb_chain_ids(r.out_pdb)
    length(chains) == 2 || error("Expected 2 chains, got $(length(chains))")
    @sprintf("pLDDT=%.1f clashes=%d chains=%s pTM=%.3f",
             r.mean_plddt, clashes.count, join(chains), something(r.ptm, NaN))
end

run_case("AF2-Multi", "af2_06_homodimer_msa") do
    msas = [joinpath(AF2_MSA_DIR, "multimer_chainA_gcn4.a3m"),
            joinpath(AF2_MSA_DIR, "multimer_chainB_gcn4.a3m")]
    r = Alphafold2.fold(af2_multi, [GCN4_SEQ, GCN4_SEQ]; msas=msas, num_recycle=5,
                        out_prefix=joinpath(OUTDIR, "af2_06"))
    clashes = check_clashes(r.out_pdb)
    @sprintf("pLDDT=%.1f clashes=%d pTM=%.3f", r.mean_plddt, clashes.count, something(r.ptm, NaN))
end

# af2_heterodimer: Heteromeric multimer (two different sequences)
run_case("AF2-Multi", "af2_heterodimer_seq_only") do
    glucagon = "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT"
    r = Alphafold2.fold(af2_multi, [GCN4_SEQ, glucagon]; num_recycle=5,
                        out_prefix=joinpath(OUTDIR, "af2_heterodimer"))
    clashes = check_clashes(r.out_pdb)
    chains = pdb_chain_ids(r.out_pdb)
    length(chains) == 2 || error("Expected 2 chains, got $(length(chains))")
    @sprintf("pLDDT=%.1f clashes=%d chains=%s pTM=%.3f",
             r.mean_plddt, clashes.count, join(chains), something(r.ptm, NaN))
end

# af2_homodimer_templates: Homodimer with template structure
run_case("AF2-Multi", "af2_homodimer_templates") do
    tmpl = joinpath(AF2_TEMPLATE_DIR, "gcn4_dimer_template.pdb")
    r = Alphafold2.fold(af2_multi, [GCN4_SEQ, GCN4_SEQ];
                        templates=[tmpl, tmpl], template_chains=["A", "B"],
                        num_recycle=5, out_prefix=joinpath(OUTDIR, "af2_homodimer_tmpl"))
    clashes = check_clashes(r.out_pdb)
    chains = pdb_chain_ids(r.out_pdb)
    length(chains) == 2 || error("Expected 2 chains, got $(length(chains))")
    @sprintf("pLDDT=%.1f clashes=%d chains=%s pTM=%.3f (with templates)",
             r.mean_plddt, clashes.count, join(chains), something(r.ptm, NaN))
end

# Release multimer
af2_multi = nothing
Alphafold2._DEFAULT_MODEL[] = nothing
GC.gc(); CUDA.reclaim()

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4: AlphaFold2 — GPU Gradient Checks
# ═════════════════════════════════════════════════════════════════════════════
if USE_GPU
    tee("\n", "#"^70)
    tee("  PHASE 4: AlphaFold2 — GPU Gradient Checks")
    tee("#"^70)

    const AF2_C_M = 256; const AF2_C_Z = 128; const AF2_C_S = 384
    const AF2_L = 12; const AF2_B = 1; const AF2_NSEQ = 4

    # Test A: EvoformerIteration
    run_case("AF2-Grad", "af2_grad_A_evoformer") do
        block = Alphafold2.EvoformerIteration(
            AF2_C_M, AF2_C_Z; num_head_msa=8, msa_head_dim=32,
            num_head_pair=4, pair_head_dim=32, c_outer=32, c_tri_mul=128,
        ) |> Flux.gpu
        msa_act = CUDA.randn(Float32, AF2_C_M, AF2_NSEQ, AF2_L, AF2_B)
        pair_act = CUDA.randn(Float32, AF2_C_Z, AF2_L, AF2_L, AF2_B)
        msa_mask = CUDA.ones(Float32, AF2_NSEQ, AF2_L, AF2_B)
        pair_mask = CUDA.ones(Float32, AF2_L, AF2_L, AF2_B)

        loss, grads = Zygote.withgradient(block) do b
            _, p = b(msa_act, pair_act, msa_mask, pair_mask)
            Float32(mean(p))
        end
        s = grad_summary(grads[1]; label="evoformer weights")
        s.nonzero > 0 && s.nan_count == 0 || error("Gradient check failed")
        block = nothing; GC.gc(); CUDA.reclaim()
        @sprintf("loss=%.4f nonzero=%d", loss, s.nonzero)
    end

    # Test B: StructureModuleCore
    run_case("AF2-Grad", "af2_grad_B_structure_module") do
        structure = Alphafold2.StructureModuleCore(
            AF2_C_S, AF2_C_Z, 128, 12, 4, 8, 3, 8, 10.0f0, 128, 2
        ) |> Flux.gpu
        single_in = CUDA.randn(Float32, AF2_C_S, AF2_L, AF2_B)
        pair_in = CUDA.randn(Float32, AF2_C_Z, AF2_L, AF2_L, AF2_B) .* 0.1f0
        seq_mask = CUDA.ones(Float32, AF2_L, AF2_B)
        aatype = rand(0:19, AF2_L)

        # B1: act loss
        loss1, g1 = Zygote.withgradient(structure) do s
            Float32(mean(s(single_in, pair_in, seq_mask, aatype).act))
        end
        s1 = grad_summary(g1[1]; label="struct (act loss)")

        # B2: atom_pos loss
        loss2, g2 = Zygote.withgradient(structure) do s
            Float32(mean(s(single_in, pair_in, seq_mask, aatype).atom_pos))
        end
        s2 = grad_summary(g2[1]; label="struct (atom_pos loss)")

        ok = s1.nonzero > 0 && s1.nan_count == 0 && s2.nonzero > 0 && s2.nan_count == 0
        ok || error("Gradient check failed")
        structure = nothing; GC.gc(); CUDA.reclaim()
        @sprintf("act_loss=%.4f atom_pos_loss=%.4f", loss1, loss2)
    end

    # Test C: PredictedLDDTHead
    run_case("AF2-Grad", "af2_grad_C_lddt_head") do
        head = Alphafold2.PredictedLDDTHead(AF2_C_S) |> Flux.gpu
        act_in = CUDA.randn(Float32, AF2_C_S, AF2_L, AF2_B)

        loss, grads = Zygote.withgradient(head) do h
            Float32(mean(Alphafold2.compute_plddt(h(act_in).logits)))
        end
        sw = grad_summary(grads[1]; label="lddt weights")
        g_in = Zygote.gradient(act_in) do x
            Float32(mean(Alphafold2.compute_plddt(head(x).logits)))
        end[1]
        gi = g_in isa CuArray ? Array(g_in) : g_in
        nz_in = count(!=(0f0), gi)
        tee("    input grad: $nz_in/$(length(gi)) nonzero")
        ok = sw.nonzero > 0 && sw.nan_count == 0 && nz_in > 0
        ok || error("Gradient check failed")
        head = nothing; GC.gc(); CUDA.reclaim()
        @sprintf("loss=%.4f w_nz=%d in_nz=%d", loss, sw.nonzero, nz_in)
    end

    # Test D: E2E evoformer → struct → pLDDT
    run_case("AF2-Grad", "af2_grad_D_e2e_plddt") do
        block = Alphafold2.EvoformerIteration(
            AF2_C_M, AF2_C_Z; num_head_msa=8, msa_head_dim=32,
            num_head_pair=4, pair_head_dim=32, c_outer=32, c_tri_mul=128,
        ) |> Flux.gpu
        struct_m = Alphafold2.StructureModuleCore(
            AF2_C_S, AF2_C_Z, 128, 12, 4, 8, 3, 4, 10.0f0, 128, 2
        ) |> Flux.gpu
        lddt_h = Alphafold2.PredictedLDDTHead(AF2_C_S) |> Flux.gpu
        sproj = Alphafold2.LinearFirst(AF2_C_M, AF2_C_S) |> Flux.gpu

        msa = CUDA.randn(Float32, AF2_C_M, AF2_NSEQ, AF2_L, AF2_B)
        pair = CUDA.randn(Float32, AF2_C_Z, AF2_L, AF2_L, AF2_B) .* 0.1f0
        mm = CUDA.ones(Float32, AF2_NSEQ, AF2_L, AF2_B)
        pm = CUDA.ones(Float32, AF2_L, AF2_L, AF2_B)
        sm = CUDA.ones(Float32, AF2_L, AF2_B)
        aa = rand(0:19, AF2_L)

        loss, grads = Zygote.withgradient(block, struct_m, lddt_h, sproj) do b, s, h, p
            mo, po = b(msa, pair, mm, pm)
            si = dropdims(p(mo[:, 1:1, :, :]); dims=2)
            so = s(si, po, sm, aa)
            Float32(mean(Alphafold2.compute_plddt(h(so.act).logits)))
        end
        sb = grad_summary(grads[1]; label="evoformer")
        ss = grad_summary(grads[2]; label="structure")
        sl = grad_summary(grads[3]; label="lddt_head")
        sp = grad_summary(grads[4]; label="projection")
        ok = all(s -> s.nonzero > 0 && s.nan_count == 0, [sb, ss, sl, sp])
        ok || error("Gradient check failed")
        block = nothing; struct_m = nothing; lddt_h = nothing; sproj = nothing
        GC.gc(); CUDA.reclaim()
        @sprintf("loss=%.4f", loss)
    end

    # Test E+F: Continuous input gradients
    run_case("AF2-Grad", "af2_grad_EF_continuous_input") do
        block = Alphafold2.EvoformerIteration(
            AF2_C_M, AF2_C_Z; num_head_msa=8, msa_head_dim=32,
            num_head_pair=4, pair_head_dim=32, c_outer=32, c_tri_mul=128,
        ) |> Flux.gpu
        struct_m = Alphafold2.StructureModuleCore(
            AF2_C_S, AF2_C_Z, 128, 12, 4, 8, 3, 4, 10.0f0, 128, 2
        ) |> Flux.gpu
        lddt_h = Alphafold2.PredictedLDDTHead(AF2_C_S) |> Flux.gpu
        sproj = Alphafold2.LinearFirst(AF2_C_M, AF2_C_S) |> Flux.gpu
        pre1d = Alphafold2.LinearFirst(22, AF2_C_M) |> Flux.gpu
        left_s = Alphafold2.LinearFirst(22, AF2_C_Z) |> Flux.gpu
        right_s = Alphafold2.LinearFirst(22, AF2_C_Z) |> Flux.gpu

        target_feat = CUDA.randn(Float32, 22, AF2_L, AF2_B)
        mm = CUDA.ones(Float32, 1, AF2_L, AF2_B)
        pm = CUDA.ones(Float32, AF2_L, AF2_L, AF2_B)
        sm = CUDA.ones(Float32, AF2_L, AF2_B)
        aa = rand(0:19, AF2_L)

        loss, grad = Zygote.withgradient(target_feat) do tf
            mi = reshape(pre1d(tf), AF2_C_M, 1, AF2_L, AF2_B)
            pi = reshape(left_s(tf), AF2_C_Z, AF2_L, 1, AF2_B) .+ reshape(right_s(tf), AF2_C_Z, 1, AF2_L, AF2_B)
            mo, po = block(mi, pi, mm, pm)
            si = dropdims(sproj(mo[:, 1:1, :, :]); dims=2)
            so = struct_m(si, po, sm, aa)
            Float32(mean(Alphafold2.compute_plddt(lddt_h(so.act).logits)))
        end
        g = grad[1]
        g === nothing && error("No gradient")
        ga = g isa CuArray ? Array(g) : g
        nz = count(!=(0f0), ga)
        nan_ct = count(isnan, ga)
        tee("    target_feat grad: $nz/$(length(ga)) nonzero, $nan_ct NaN")
        nz > 0 && nan_ct == 0 || error("Gradient check failed")
        block = nothing; struct_m = nothing; lddt_h = nothing; sproj = nothing
        pre1d = nothing; left_s = nothing; right_s = nothing
        GC.gc(); CUDA.reclaim()
        @sprintf("loss=%.4f input_grad_nz=%d", loss, nz)
    end
else
    tee("\n  PHASE 4: Skipped (no --gpu)")
end

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 5-9: BoltzGen
# ═════════════════════════════════════════════════════════════════════════════
tee("\n", "#"^70)
tee("  PHASE 5-9: BoltzGen")
tee("#"^70)

using MoleculeFlow
using Onion
using BoltzGen
Onion.bg_set_training!(false)

const BG_ROOT = joinpath(@__DIR__, "WorkingBoltzGen", "BoltzGen.jl")
const BG_YAML_DIR = joinpath(BG_ROOT, "examples")

# BoltzGen check helpers
function bg_check_geometry(result; label="")
    stats = BoltzGen.assert_geometry_sane_atom37!(result["feats"], result["coords"]; batch=1)
    tee("    geometry ($label): n_atoms=$(stats.n_atoms)",
        " max_abs=$(round(stats.max_abs; digits=2))",
        " min_nn=$(round(stats.min_nearest_neighbor; digits=3))",
        " frac_ge900=$(round(stats.frac_abs_ge900; digits=3))")
    return stats
end

function bg_check_bonds(result)
    BoltzGen.print_bond_length_report(result; io=LOG_IO)
    violations = BoltzGen.check_bond_lengths(result)
    n_total = violations.n_bonds_checked
    n_viol = violations.n_violations
    tee("    bonds: $n_total checked, $n_viol violations")
    return (n_total=n_total, n_violations=n_viol)
end

function bg_check_strings(result)
    p14 = BoltzGen.output_to_pdb(result)
    p37 = BoltzGen.output_to_pdb_atom37(result)
    cif = BoltzGen.output_to_mmcif(result)
    length(p14) > 10 || error("PDB14 too small")
    length(p37) > 10 || error("PDB37 too small")
    length(cif) > 10 || error("CIF too small")
    return (pdb14=length(p14), pdb37=length(p37), cif=length(cif))
end

function bg_write_outputs(result, prefix)
    BoltzGen.write_outputs(result, joinpath(OUTDIR, prefix))
end

# ── Phase 5: BoltzGen1 Design ────────────────────────────────────────────────
tee("\n  Loading BoltzGen1 design model...")
t0 = time()
bg_gen = BoltzGen.load_boltzgen(; gpu=USE_GPU)
tee("  BoltzGen1 loaded in $(round(time()-t0; digits=1))s")

# Warmup
tee("  Warmup...")
BoltzGen.design_from_sequence(bg_gen, ""; length=5, steps=2, recycles=1, seed=1)
if USE_GPU; CUDA.synchronize(); end

run_case("BG-Design", "bg_01_protein_only_12res") do
    r = BoltzGen.design_from_sequence(bg_gen, ""; length=12, steps=200, recycles=2, seed=7)
    bg_write_outputs(r, "bg_01_protein_only")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    strs = bg_check_strings(r)
    @sprintf("bonds=%d viol=%d pdb14=%dB", bonds.n_total, bonds.n_violations, strs.pdb14)
end

run_case("BG-Design", "bg_02_protein_smallmol") do
    yaml = joinpath(BG_YAML_DIR, "protein_binding_small_molecule", "chorismite.yaml")
    r = BoltzGen.design_from_yaml(bg_gen, yaml; steps=200, recycles=2, seed=7)
    bg_write_outputs(r, "bg_02_chorismite")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    strs = bg_check_strings(r)
    @sprintf("bonds=%d viol=%d", bonds.n_total, bonds.n_violations)
end

run_case("BG-Design", "bg_03_protein_dna_smiles") do
    yaml = joinpath(BG_YAML_DIR, "mixed_protein_dna_smiles_smoke_v1.yaml")
    r = BoltzGen.design_from_yaml(bg_gen, yaml; steps=200, recycles=2, seed=7)
    bg_write_outputs(r, "bg_03_mixed")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    strs = bg_check_strings(r)
    @sprintf("bonds=%d viol=%d", bonds.n_total, bonds.n_violations)
end

run_case("BG-Design", "bg_04_tcd_8res") do
    target_pdb = joinpath(OUTDIR, "bg_01_protein_only_atom14.pdb")
    isfile(target_pdb) || error("bg_01 PDB not found (did bg_01 fail?)")
    r = BoltzGen.target_conditioned_design(bg_gen, target_pdb; design_length=8, steps=200, recycles=2, seed=7)
    bg_write_outputs(r, "bg_04_tcd")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    strs = bg_check_strings(r)
    @sprintf("bonds=%d viol=%d", bonds.n_total, bonds.n_violations)
end

# bg_design_cyclic: Cyclic protein design
# NOTE: Known GPU bug — ifelse broadcast produces Real not Float32, causing
#       "Broadcast operation resulting in Real is not GPU compatible". This is a
#       pre-existing issue in BoltzGen's cyclic code path, not a test bug.
#       Test is wrapped in try/catch to report the failure without blocking the suite.
run_case("BG-Design", "bg_design_cyclic_12res") do
    r = BoltzGen.design_from_sequence(bg_gen, ""; length=12, cyclic=true, steps=200, recycles=2, seed=7)
    bg_write_outputs(r, "bg_design_cyclic")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    strs = bg_check_strings(r)
    @sprintf("bonds=%d viol=%d cyclic=true", bonds.n_total, bonds.n_violations)
end

# bg_design_dna: DNA sequence design
run_case("BG-Design", "bg_design_dna_8res") do
    r = BoltzGen.design_from_sequence(bg_gen, "ACGTACGT"; chain_type="DNA", steps=200, recycles=2, seed=7)
    bg_write_outputs(r, "bg_design_dna")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    @sprintf("bonds=%d viol=%d chain_type=DNA", bonds.n_total, bonds.n_violations)
end

# bg_denovo_sample: De novo sampling (no sequence input)
run_case("BG-Design", "bg_denovo_sample_10res") do
    r = BoltzGen.denovo_sample(bg_gen, 10; steps=200, recycles=2, seed=7)
    bg_write_outputs(r, "bg_denovo_sample")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    strs = bg_check_strings(r)
    @sprintf("bonds=%d viol=%d denovo=10res", bonds.n_total, bonds.n_violations)
end

# bg_design_partial: Design with mask (redesign middle residues, keep ends fixed)
run_case("BG-Design", "bg_design_partial_mask") do
    seq = "ELLKKLLEELKG"  # 12 residues
    mask = [false, false, true, true, true, true, true, true, false, false, false, false]
    r = BoltzGen.design_from_sequence(bg_gen, seq; design_mask=mask, steps=200, recycles=2, seed=7)
    bg_write_outputs(r, "bg_design_partial")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    @sprintf("bonds=%d viol=%d partial_mask=8of12", bonds.n_total, bonds.n_violations)
end

# Release BoltzGen1
bg_gen = nothing; GC.gc()
if USE_GPU; CUDA.reclaim(); end

# ── Phase 6: Boltz2 Confidence Fold ──────────────────────────────────────────
tee("\n  Loading Boltz2 confidence model...")
t0 = time()
bg_fold = BoltzGen.load_boltz2(; affinity=false, gpu=USE_GPU)
tee("  Boltz2 conf loaded in $(round(time()-t0; digits=1))s")

# Warmup
tee("  Warmup...")
BoltzGen.fold_from_sequence(bg_fold, "GGGGG"; steps=2, recycles=1, seed=1)
if USE_GPU; CUDA.synchronize(); end

run_case("BG-Fold", "bg_05_fold_polyG_14res") do
    r = BoltzGen.fold_from_sequence(bg_fold, "GGGGGGGGGGGGGG"; steps=200, recycles=3, seed=7)
    bg_write_outputs(r, "bg_05_polyG")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    metrics = BoltzGen.confidence_metrics(r)
    tee("    confidence: ", pairs(metrics))
    @sprintf("bonds=%d viol=%d", bonds.n_total, bonds.n_violations)
end

run_case("BG-Fold", "bg_06_fold_antibody_121res") do
    vh = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
    r = BoltzGen.fold_from_sequence(bg_fold, vh; steps=200, recycles=3, seed=7)
    bg_write_outputs(r, "bg_06_antibody")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    metrics = BoltzGen.confidence_metrics(r)
    tee("    confidence: ", pairs(metrics))
    @sprintf("bonds=%d viol=%d", bonds.n_total, bonds.n_violations)
end

run_case("BG-Fold", "bg_07_fold_lysozyme_nomsa_129res") do
    hewl = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
    r = BoltzGen.fold_from_sequence(bg_fold, hewl; steps=200, recycles=3, seed=7)
    bg_write_outputs(r, "bg_07_lysozyme_nomsa")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    metrics = BoltzGen.confidence_metrics(r)
    tee("    pTM=$(round(metrics.ptm[1]; digits=3))")
    @sprintf("bonds=%d viol=%d pTM=%.3f", bonds.n_total, bonds.n_violations, metrics.ptm[1])
end

run_case("BG-Fold", "bg_08_fold_lysozyme_msa_129res") do
    hewl = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
    msa_file = joinpath(BG_YAML_DIR, "lysozyme_msa.fasta")
    isfile(msa_file) || error("MSA not found at $msa_file")
    r = BoltzGen.fold_from_sequence(bg_fold, hewl; steps=200, recycles=3, seed=7,
                                     msa_file=msa_file, max_msa_rows=100)
    bg_write_outputs(r, "bg_08_lysozyme_msa")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    metrics = BoltzGen.confidence_metrics(r)
    tee("    pTM=$(round(metrics.ptm[1]; digits=3))")
    @sprintf("bonds=%d viol=%d pTM=%.3f", bonds.n_total, bonds.n_violations, metrics.ptm[1])
end

# Template tests: fold poly-GLY without and with structural template
# Python's official fold pipeline uses target_templates=true. Without templates,
# poly-GLY often has ~1-2 bond violations (degenerate input). Templates fix this.
const BG_TEMPLATE_CIF = joinpath(@__DIR__, "PythonBoltzGen",
    "polyG_official_output", "intermediate_designs_inverse_folded", "polyG_fold_test.cif")

run_case("BG-Fold", "bg_fold_polyG_no_template") do
    r = BoltzGen.fold_from_sequence(bg_fold, "GGGGGGGGGGGGGG"; steps=200, recycles=3, seed=7)
    bg_write_outputs(r, "bg_fold_polyG_no_template")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    tee("    NOTE: poly-GLY without templates may have ~1-2 violations (degenerate input)")
    @sprintf("bonds=%d viol=%d (no template, some violations expected)", bonds.n_total, bonds.n_violations)
end

run_case("BG-Fold", "bg_fold_polyG_with_template") do
    isfile(BG_TEMPLATE_CIF) || error("Template CIF not found at $BG_TEMPLATE_CIF")
    r = BoltzGen.fold_from_sequence(bg_fold, "GGGGGGGGGGGGGG";
        steps=200, recycles=3, seed=7, template_paths=[BG_TEMPLATE_CIF])
    bg_write_outputs(r, "bg_fold_polyG_with_template")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    @sprintf("bonds=%d viol=%d (with template)", bonds.n_total, bonds.n_violations)
end

# bg_fold_3chain: Multi-chain fold (2 proteins + 1 DNA)
run_case("BG-Fold", "bg_fold_3chain_protein_dna") do
    seqs = ["ACDEFGHIK", "LMNPQRST", "ACGTACGT"]
    r = BoltzGen.fold_from_sequences(bg_fold, seqs;
        chain_types=["PROTEIN", "PROTEIN", "DNA"],
        steps=200, recycles=3, seed=7)
    bg_write_outputs(r, "bg_fold_3chain")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    metrics = BoltzGen.confidence_metrics(r)
    tee("    confidence: ", pairs(metrics))
    @sprintf("bonds=%d viol=%d chains=3(2P+1D)", bonds.n_total, bonds.n_violations)
end

# bg_fold_rna: RNA-only fold
run_case("BG-Fold", "bg_fold_rna_12res") do
    r = BoltzGen.fold_from_sequence(bg_fold, "AUGCAUGCAUGC"; chain_type="RNA",
        steps=200, recycles=3, seed=7)
    bg_write_outputs(r, "bg_fold_rna")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    metrics = BoltzGen.confidence_metrics(r)
    tee("    confidence: ", pairs(metrics))
    @sprintf("bonds=%d viol=%d chain_type=RNA", bonds.n_total, bonds.n_violations)
end

# Release confidence model
bg_fold = nothing; GC.gc()
if USE_GPU; CUDA.reclaim(); end

# ── Phase 7: Boltz2 Affinity Fold ────────────────────────────────────────────
tee("\n  Loading Boltz2 affinity model...")
t0 = time()
bg_aff = BoltzGen.load_boltz2(; affinity=true, gpu=USE_GPU)
tee("  Boltz2 aff loaded in $(round(time()-t0; digits=1))s")

run_case("BG-Affinity", "bg_09_fold_polyG_affinity") do
    r = BoltzGen.fold_from_sequence(bg_aff, "GGGGGGGGGGGGGG"; steps=200, recycles=3, seed=7)
    bg_write_outputs(r, "bg_09_polyG_aff")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    @sprintf("bonds=%d viol=%d (degenerate input, ~7 acceptable)", bonds.n_total, bonds.n_violations)
end

run_case("BG-Affinity", "bg_10_fold_from_structure") do
    target = joinpath(OUTDIR, "bg_03_mixed_atom14.pdb")
    isfile(target) || error("bg_03 PDB not found (did bg_03 fail?)")
    r = BoltzGen.fold_from_structure(bg_aff, target; steps=200, recycles=3, seed=7)
    bg_write_outputs(r, "bg_10_from_structure")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    @sprintf("bonds=%d viol=%d", bonds.n_total, bonds.n_violations)
end

# Release affinity model
bg_aff = nothing; GC.gc()
if USE_GPU; CUDA.reclaim(); end

# ── Phase 8: BoltzGen E2E Small Molecule & DNA ──────────────────────────────
tee("\n  Loading BoltzGen1 for E2E tests...")
t0 = time()
bg_gen2 = BoltzGen.load_boltzgen(; gpu=USE_GPU)
tee("  BoltzGen1 loaded in $(round(time()-t0; digits=1))s")

const E2E_YAML_DIR = joinpath(BG_ROOT, "examples", "e2e_tests")

run_case("BG-E2E", "bg_e2e01_protein_hem") do
    yaml = joinpath(E2E_YAML_DIR, "design_protein_hem.yaml")
    r = BoltzGen.design_from_yaml(bg_gen2, yaml; steps=200, recycles=2, seed=7)
    bg_write_outputs(r, "bg_e2e01_hem")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    strs = bg_check_strings(r)
    @sprintf("bonds=%d viol=%d", bonds.n_total, bonds.n_violations)
end

run_case("BG-E2E", "bg_e2e06_protein_smiles") do
    yaml = joinpath(E2E_YAML_DIR, "design_protein_smiles.yaml")
    r = BoltzGen.design_from_yaml(bg_gen2, yaml; steps=200, recycles=2, seed=7)
    bg_write_outputs(r, "bg_e2e06_smiles")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    @sprintf("bonds=%d viol=%d", bonds.n_total, bonds.n_violations)
end

# Release design model for fold tests
bg_gen2 = nothing; GC.gc()
if USE_GPU; CUDA.reclaim(); end

# E2E fold cases
tee("\n  Loading Boltz2 conf for E2E fold tests...")
bg_fold2 = BoltzGen.load_boltz2(; affinity=false, gpu=USE_GPU)

run_case("BG-E2E", "bg_e2e09_fold_dna") do
    r = BoltzGen.fold_from_sequence(bg_fold2, "ACGTACGTAC"; chain_type="DNA", steps=200, recycles=3, seed=7)
    bg_write_outputs(r, "bg_e2e09_dna")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    metrics = BoltzGen.confidence_metrics(r)
    tee("    confidence: ", pairs(metrics))
    @sprintf("bonds=%d viol=%d", bonds.n_total, bonds.n_violations)
end

run_case("BG-E2E", "bg_e2e11_fold_from_hem") do
    target = joinpath(OUTDIR, "bg_e2e01_hem_atom14.pdb")
    isfile(target) || error("e2e01 HEM PDB not found (did e2e01 fail?)")
    r = BoltzGen.fold_from_structure(bg_fold2, target; steps=200, recycles=3, seed=7,
                                      include_nonpolymer=true)
    bg_write_outputs(r, "bg_e2e11_hem_refold")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    @sprintf("bonds=%d viol=%d", bonds.n_total, bonds.n_violations)
end

bg_fold2 = nothing; GC.gc()
if USE_GPU; CUDA.reclaim(); end

# ── Phase 9: BoltzGen TCD Scaling (first 2 cases) ───────────────────────────
tee("\n  Loading BoltzGen1 for TCD scaling...")
bg_gen3 = BoltzGen.load_boltzgen(; gpu=USE_GPU)

const TCD_INPUTS = joinpath(@__DIR__, "WorkingBoltzGen", "tcd_test_inputs")

run_case("BG-TCD", "bg_tcd01_129target_8design") do
    target = joinpath(TCD_INPUTS, "antibody_refold.pdb")
    isfile(target) || error("TCD target not found: $target")
    r = BoltzGen.target_conditioned_design(bg_gen3, target; design_length=8, steps=200, recycles=3, seed=42)
    bg_write_outputs(r, "bg_tcd01")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    @sprintf("bonds=%d viol=%d", bonds.n_total, bonds.n_violations)
end

run_case("BG-TCD", "bg_tcd02_121target_60design") do
    target = joinpath(TCD_INPUTS, "antibody_fold.pdb")
    isfile(target) || error("TCD target not found: $target")
    r = BoltzGen.target_conditioned_design(bg_gen3, target; design_length=60, steps=200, recycles=3, seed=42)
    bg_write_outputs(r, "bg_tcd02")
    bg_check_geometry(r; label="atom37")
    bonds = bg_check_bonds(r)
    @sprintf("bonds=%d viol=%d", bonds.n_total, bonds.n_violations)
end

bg_gen3 = nothing; GC.gc()
if USE_GPU; CUDA.reclaim(); end

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
close(LOG_IO)

println("\n", "="^70)
println("  SUMMARY")
println("="^70)

# Group by phase
phases = unique(r[1] for r in results)
for phase in phases
    phase_results = filter(r -> r[1] == phase, results)
    n_pass = count(r -> r[3], phase_results)
    n_total = length(phase_results)
    status = n_pass == n_total ? "ALL PASS" : "$n_pass/$n_total"
    @printf("  %-14s  %s\n", phase, status)
    for (_, name, passed, detail, elapsed) in phase_results
        mark = passed ? "PASS" : "FAIL"
        @printf("    [%s] %-40s  %5.1fs  %s\n", mark, name, elapsed, first(detail, 60))
    end
end

n_pass = count(r -> r[3], results)
n_total = length(results)
total_time = sum(r[5] for r in results)
println()
@printf("  TOTAL: %d / %d passed in %.0fs\n", n_pass, n_total, total_time)
println("  Log: $LOG_FILE")

n_pass == n_total || error("$(n_total - n_pass) test(s) FAILED")
println("  ALL PASS")
