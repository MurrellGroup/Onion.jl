# Primitives Gap Analysis: Onion (primitives branch) vs Model Requirements

## Overview

This document maps what the three downstream models (AlphaFold2, ESMFold, BoltzGen) need
against what the new Onion primitives system provides, identifying gaps.

---

## 1. Current Primitives (Onion, `primitives` branch)

Six primitives declared in `src/primitives/primitives.jl`:

| Primitive | Signature | Default Backend | cuTile Backend | Used By Layer |
|-----------|-----------|-----------------|----------------|---------------|
| `rms_norm` | `(x, w; eps, offset)` | lazy broadcast | 2-kernel fwd + 2-kernel bwd | `RMSNorm` |
| `layer_norm` | `(x, w, b; eps)` | mean/var stats | 2-kernel fwd + 2-kernel bwd | `LayerNorm` |
| `softmax` | `(x, dims=1)` | NNlib.softmax | online 2-pass | (internal to attention) |
| `attention` | `(q,k,v; pair, kpad_mask, k_lengths, causal)` | matmul+softmax | flash attention | `Attention` |
| `glu_ffn` | `(x, W_gate, W_up, W_down, act)` | 3 matmuls | 2-fwd + 4-bwd kernels | `StarGLU` |
| `multihead_ffn` | `(Q, Kg_T, Ku_T, V, act)` | einsum | 1-fwd + 2-bwd kernels | (none yet) |

### cuTile attention features
- Pair bias (additive attention bias from pairwise state)
- `k_lengths` (per-batch variable key sequence lengths)
- `q_lengths` (per-batch variable query lengths, for future use)
- Causal masking
- Grouped query attention (`query_group_size > 1`)
- Forward: stores M (max) and L (normalizer) for backward recomputation
- Full backward: dQ, dK, dV, dBias

---

## 2. What Each Model Needs

### 2.1 AlphaFold2.jl — Direct Onion Primitive Calls

| Call | Where | Tensor Layout | Maps To |
|------|-------|---------------|---------|
| `flash_attention_bias_forward(q4, k4, v4, bias)` | attention.jl:121 | (D,seq,H,B) + (K,Q,H,1) | `attention` with `pair=bias` |
| `flash_attention_forward(q4, k4, v4)` | attention.jl:123 | (D,seq,H,B) | `attention` (no bias) |
| `combine_projections_forward(left, right, outgoing)` | triangle.jl:46 | (C,L,L,B) | **MISSING primitive** |
| `layernorm_first_forward(x, w, b; eps)` | layers.jl:18 | (in_dim,...) | `layer_norm` |
| Rigid ops (20+ functions) | rigid.jl | various | Domain types (not primitives) |
| `InvariantPointAttention` | fold_iteration_core.jl | (c_s,L,B) pairs | Layer (not primitive) |

### 2.2 ESMFold.jl — Direct Onion Primitive Calls

| Call | Where | Tensor Layout | Maps To |
|------|-------|---------------|---------|
| `flash_attention_forward(q, k, v)` | via ESMMultiheadAttention | (D,L,H,B) | `attention` |
| `flash_attention_bias_forward(q, k, v, bias)` | via ESMFoldAttention | (D,L,H,B) + bias | `attention` with `pair=bias` |
| `layernorm_first_forward(x, w, b; eps)` | via LayerNormFirst | (C,...) | `layer_norm` |
| `rotary_pos_emb_forward(x, cos, sin)` | via ESMMultiheadAttention | (D,L,H,B) | **MISSING primitive** |
| `combine_projections_forward(a, b, outgoing)` | via TriangleMultiplicativeUpdate | (C,L,L,B) | **MISSING primitive** |
| Rigid ops, StructureModule, IPA | structure_module.jl | various | Domain types/layers |

### 2.3 BoltzGen.jl — Onion Layer/Primitive Usage

| Usage | Where | Maps To |
|-------|-------|---------|
| `Onion.AttentionPairBias(dim, dim, heads)` | transformers.jl:90 | Layer using `attention` + `layer_norm` |
| `Onion.PairformerModule(...)` | boltz.jl:174 | Composite layer (triangle ops, attention, transitions) |
| `Onion.PairformerNoSeqModule(...)` | trunk.jl:305 | Composite layer (pair-only) |
| `Onion.MiniformerNoSeqModule(...)` | trunk.jl:303 | Lightweight pairformer variant |
| `Onion.Transition(dim, hidden)` | encoders.jl:144 | FFN layer → could use `glu_ffn` |
| `Onion.OuterProductMean(...)` | trunk.jl:518 | Layer (batched_mul, not primitive) |
| `Onion.PairWeightedAveraging(...)` | trunk.jl:512 | Layer (batched_mul + softmax) |
| `Onion.BGLayerNorm(dim)` | everywhere | `layer_norm` |
| `Onion.BGLinear(in, out)` | everywhere | Dense (matmul, not primitive) |
| Init functions (`torch_linear_init!`, etc.) | throughout | Utilities (not primitives) |

---

## 3. Gap Analysis

### 3.1 MISSING Primitives (needed by models, not in new Onion)

#### `rotary_pos_emb` — Rotary Positional Embeddings
- **Used by:** ESMFold (ESM2 transformer, 33 layers × every forward pass)
- **Old Onion-2:** `rotary_pos_emb_forward(x, cos, sin)` dispatch hook
- **OnionTile:** `apply_rotary_pos_emb_fused(x, cos, sin)` — 2 broadcasts + vcat
- **cuTileExt:** Not yet implemented
- **Signature:** `rotary_pos_emb(x, cos, sin)` where x is (D, L, H, B)
- **Default impl:** `x .* cos .+ rotate_half(x) .* sin`
- **Priority:** HIGH — called 33× per ESM2 forward pass

#### `combine_projections` — Triangle Multiplication Contraction
- **Used by:** AlphaFold2 (evoformer), ESMFold (folding trunk), BoltzGen (pairformer)
- **Old Onion-2:** `combine_projections_forward(a, b, outgoing)` dispatch hook
- **OnionTile:** `cutensor_combine_projections(a, b, outgoing)` — cuTENSOR cached plans
- **cuTileExt:** Not yet implemented
- **Signature:** `combine_projections(a, b, outgoing::Bool)` where a,b are (C, L, L, B)
- **Default impl:** `NNlib.batched_mul` with appropriate transpose
- **Priority:** HIGH — used in every evoformer/pairformer iteration

### 3.2 Existing Primitives — Mapping Verification

#### `attention` — GOOD MATCH
The new `attention` primitive supports:
- `pair` kwarg → maps to AF2/ESMFold `flash_attention_bias_forward`
- No `pair` → maps to `flash_attention_forward`
- `k_lengths` → useful for BoltzGen variable-length sequences
- cuTile backend already implements all of this

**Layout consideration:** All models use (D, L, H, B) layout. cuTileExt already uses this.

**Action needed:** Verify that the old `flash_attention_bias_forward` bias layout (K, Q, H, B)
matches the new `attention` primitive's `pair` kwarg layout. The old Onion-2 `gpu_dispatch.jl`
permutes bias from (K, Q, H, B) to the format flash attention expects.

#### `layer_norm` — GOOD MATCH
- All three models use LayerNormFirst which calls `layernorm_first_forward`
- The new `layer_norm` primitive signature matches: `(x, w, b; eps)`
- cuTileExt already implements this with ChainRules rrule

**Note:** BoltzGen aliases `BGLayerNorm = LayerNormFirst`. This should continue to work.

#### `rms_norm` — NOT NEEDED by protein models
- None of the three models use RMSNorm (they all use LayerNorm)
- Useful for LLM workloads (LLaMA, etc.) but not protein models
- Still valuable to keep as a primitive

#### `softmax` — INTERNAL ONLY
- Used internally by the attention primitive's default implementation
- Models don't call softmax directly (it's inside attention)
- cuTile attention handles softmax internally (online softmax)

#### `glu_ffn` — PARTIAL MATCH
- StarGLU layer uses this for SwiGLU feed-forward networks
- BoltzGen `Transition` layer uses LN → FC1(GELU) → FC2 → FC3 (NOT SwiGLU)
- AF2/ESMFold transitions use LN → Linear(ReLU) → Linear (simple MLP, NOT gated)
- So `glu_ffn` is useful for LLM workloads but protein models mostly use simpler FFNs

**Note:** The old Onion-2 `Transition` in boltzgen/ is: `LN → Linear(SiLU) → Linear → Linear`
which IS a gated structure but with 3 projections, not the standard SwiGLU 2-gate pattern.

#### `multihead_ffn` — NOT YET USED
- Defined but not integrated into any layer
- cuTile binding marked XXX for correctness verification
- Not needed by any protein model currently

### 3.3 Layers Needed (not primitives, but must exist in Onion)

These are composite layers that the models import from Onion. They should be rebuilt
in the new Onion using the new primitives system:

#### Protein-Universal Layers (used by 2+ models)
- `LayerNormFirst` → wraps `layer_norm` primitive
- `LinearFirst` → matmul (no primitive needed)
- `TriangleMultiplicativeUpdate` → uses `combine_projections` primitive + gating
- `TriangleMultiplicationOutgoing/Incoming` → convenience wrappers
- `TriangleAttention` → uses `attention` primitive with pair bias
- `StructureModule` → uses IPA + backbone + angles
- `InvariantPointAttention` / `ESMFoldIPA` → augmented attention (scalar + point features)
- `BackboneUpdate`, `AngleResnet`, `StructureModuleTransition` → simple layers

#### ESMFold-Specific Layers
- `ESMMultiheadAttention` → `attention` + `rotary_pos_emb` primitives
- `TriangularSelfAttentionBlock` → composite of all triangle ops + attention
- `ESMFoldAttention` → attention with pair bias + optional gating
- `FoldingTrunk` → orchestrator for all trunk blocks
- `SequenceToPair`, `PairToSequence`, `ResidueMLP`

#### BoltzGen-Specific Layers
- `AttentionPairBias` → `attention` + `layer_norm` + optional Q/K norm
- `PairformerModule/Layer` → triangle ops + attention + transitions
- `MiniformerModule/Layer` → lightweight pairformer variant
- `Transition` → FFN (LN → Linear(SiLU) → Linear → Linear)
- `OuterProductMean` → batched_mul contraction
- `PairWeightedAveraging` → batched_mul + softmax
- `BGLayerNorm` → alias for LayerNormFirst
- `BGLinear` → LinearFirst with init schemes

#### Domain Types (protein-specific)
- `Rigid`, `RotMatRotation`, `QuatRotation` — SE(3) transforms
- All rigid body operations (compose, apply, invert, etc.)
- Residue constants (atom types, frames, literature positions)
- `atom14_to_atom37`, `torsion_angles_to_frames`, etc.

### 3.4 OnionTile Kernels vs cuTileExt Coverage

| OnionTile Kernel | cuTileExt Equivalent | Gap? |
|-----------------|---------------------|------|
| `flash_attention` (no bias) | `attention` primitive (cuTile) | Covered |
| `flash_attention_bias` | `attention` primitive with `pair` | Covered |
| `flash_attention_train` + backward | ChainRules rrule in cuTileExt | Covered |
| `fused_layernorm_first` | `layer_norm` primitive (cuTile) | Covered |
| `fused_layernorm_first!` (in-place) | Not in cuTileExt | Minor gap |
| `apply_rotary_pos_emb_fused` | **NOT in cuTileExt** | **GAP** |
| `cutensor_combine_projections` | **NOT in cuTileExt** | **GAP** |
| `fused_linear_forward` | Not a primitive | Minor gap |
| `safe_checkpointed` | Not a primitive | Utility |
| `esm2_flash_attention` (3D→4D) | Not needed (reshape at call site) | N/A |

---

## 4. Recommended Actions

### Priority 1: Add Missing Primitives

#### 4.1 `rotary_pos_emb` primitive
```julia
@primitive rotary_pos_emb
# Default: x .* cos .+ rotate_half(x) .* sin
# cuTile: 2 fused broadcasts + vcat (from OnionTile)
```

#### 4.2 `combine_projections` primitive
```julia
@primitive combine_projections
# Default: NNlib.batched_mul with transpose logic
# cuTile: cuTENSOR cached contraction plans (from OnionTile)
```

### Priority 2: Port Layer Types to New Architecture

The old Onion-2 `src/protein/` module has ~4000 lines of layer definitions.
These need to be ported to use the new primitives system. Strategy:

1. **Core layers** (used by all models): Port first
   - `LayerNormFirst` → `LayerNorm` (already exists in new Onion)
   - `LinearFirst` → simple layer, minimal change needed
   - `TriangleMultiplicativeUpdate` → uses `combine_projections` primitive
   - `TriangleAttention` → uses `attention` primitive with bias

2. **Model-specific layers**: Port per-model as needed
   - ESMFold layers (ESMMultiheadAttention, FoldingTrunk, etc.)
   - BoltzGen layers (Pairformer, AttentionPairBias, etc.)
   - AF2 layers (EvoformerIteration, StructureModuleCore, etc.)

3. **Domain types**: Port as-is (rigid body ops, protein constants)
   - These are not primitives, just data types and pure functions
   - Minimal refactoring needed

### Priority 3: cuTileExt Gaps

1. Add `rotary_pos_emb` binding (port from OnionTile)
2. Add `combine_projections` binding (port from OnionTile, needs cuTENSOR)
3. Consider `fused_linear_forward` as optimization (not a primitive)

### Priority 4: Verify Tensor Layouts

The new attention primitive and old dispatch hooks may differ in bias layout:
- Old: bias shape (K, Q, H, B), permuted to (H, Q, K, B) in gpu_dispatch.jl
- New: `pair` kwarg — need to verify expected shape
- Action: Read cuTileExt attention binding to confirm layout expectations

---

## 5. Simplification Opportunities

### 5.1 Unified Attention
The old Onion-2 had separate dispatch hooks:
- `flash_attention_forward` (no bias)
- `flash_attention_bias_forward` (with bias)

The new `attention` primitive unifies these with `pair=nothing` default.
This is cleaner — models just pass `pair=bias` when they have one.

### 5.2 Eliminate GPU Dispatch Hooks
The old system used function-based dispatch (`layernorm_first_forward` etc.)
with CuArray method overrides. The new system uses Backend dispatch, which is
cleaner and more extensible. All gpu_dispatch.jl can be eliminated.

### 5.3 Eliminate gpu_layers.jl In-Place Hacks
The old system had extensive in-place operation hacks controlled by
`within_gradient()`. With proper primitives + cuTile kernels, the backend
handles in-place vs allocating internally. Layers become simpler.

### 5.4 Consolidate BoltzGen Layer Aliases
`BGLayerNorm = LayerNormFirst`, `BGLinear = LinearFirst` — these aliases
exist for historical reasons. In the new Onion, we should just use the
canonical names everywhere.

### 5.5 Consolidate Triangle Operations
Both ESMFold and BoltzGen define similar triangle operations differently.
With `combine_projections` as a primitive, all triangle layers can use the
same primitive call.

---

## 6. Kernel Count Summary

| Primitive | Default Kernels | cuTile Kernels | Total GPU Kernels |
|-----------|----------------|----------------|-------------------|
| rms_norm | 1 (broadcast) | 4 (fwd+bwd) | 4 |
| layer_norm | 1 (broadcast) | 4 (fwd+bwd) | 4 |
| softmax | 1 (NNlib) | 2 (fwd+bwd) | 2 |
| attention | ~5 (matmul+softmax) | 3+ (flash fwd+bwd) | 3+ |
| glu_ffn | 3 (matmuls) | 6 (2-fwd+4-bwd) | 6 |
| multihead_ffn | 2 (einsum) | 3 (1-fwd+2-bwd) | 3 |
| rotary_pos_emb (NEW) | 1 (broadcast) | 2 (fused broadcasts) | 2 |
| combine_projections (NEW) | 1 (batched_mul) | 2 (cuTENSOR fwd+bwd) | 2 |
| **TOTAL** | ~15 | **~26** | |
