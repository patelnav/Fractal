# Plan: Fix Phase 31 Generation (Target: 85%+)

## Current State

| Model | Generation | Has = | Repair | Editing | Notes |
|-------|------------|-------|--------|---------|-------|
| V1 + Hybrid AR3 | **57.0%** | 10% | 84.5% | 100% | **BEST** - AR warmstart + MaskGIT |
| V1 (baseline) | 38.5% | 20% | 83.8% | 100% | σ ∈ [0.1, 0.9], 20K samples, 2000 iters |
| V4 (self-cond) | 25.0% | 16% | 83.4% | 100% | Self-conditioning FAILED |
| V2 (scaled up) | 39% | 33% | 84% | 100% | σ ∈ [0.1, 1.0] + 30% high-noise bias - FAILED |
| V3 (minimal) | 43% | 1% | TBD | TBD | σ ∈ [0.1, 1.0] - FAILED |

**Current Best:** 57% syntax validity with Hybrid AR3+MaskGIT (target: 85%)
**Improvement:** +18.5pp over V1 MaskGIT baseline (38.5% → 57%)

**Original Hypothesis (WRONG):** Training distribution mismatch (σ ∈ [0.1, 0.9], never sees σ = 1.0)

**Revised Understanding:** Extending σ_max to 1.0 makes things WORSE, not better:
- V2: Generation dropped 51% → 39%
- V3: Generation dropped 53.5% → 43%, and '=' almost never appears (1%)

**Why σ=1.0 hurts:** At σ=1.0, ALL tokens become MASK. The model sees (all MASK) → (clean) pairs with no conditional signal. This may teach the model to ignore the input entirely, which breaks the iterative refinement that MaskGIT depends on.

---

## Next Steps: Alternative Approaches

Since extending σ_max failed, we need different strategies:

### Option A: Self-Conditioning (from FIX_GENERATION.md Tier 2)
Feed previous iteration's prediction as auxiliary input. Proven on Imagen, SoundStorm.

```python
# In forward pass
if self.prev_logits is not None:
    x_prev_estimate = self.prev_logits.argmax(-1)
    hidden = self.embed(x_noisy) + 0.5 * self.embed(x_prev_estimate) + self.noise_embed(σ)
```

**Effort:** ~2 hours to implement and test
**Rationale:** Gives the model "memory" of its previous guess, reducing the cold-start problem

### Option B: AR Warmstart (Hybrid Approach)
Generate first 3-5 tokens autoregressively, then switch to bidirectional fill.

```python
def generate_hybrid(model, length):
    # Phase 1: AR generates initial structure
    x = [BOS_ID]
    for _ in range(3):
        logits = causal_model(x)  # Need a causal head
        x.append(sample(logits[-1]))

    # Phase 2: Pad with MASK, run bidirectional
    x = x + [MASK_ID] * (length - len(x) - 1) + [EOS_ID]
    return maskgit_refine(model, x)
```

**Effort:** ~4 hours (need causal attention head)
**Rationale:** Provides structural anchors naturally

### Option C: Two-Stage: Coarse → Fine
1. First model generates "skeleton" (operators, parens, =)
2. Second pass fills in digits

**Effort:** ~1 day
**Rationale:** Separates structural decisions from content

---

## Experiment Results

### Hyperparameter Experiments (Failed)

| Version | Config | Generation | Has = | Status |
|---------|--------|------------|-------|--------|
| V1 | σ_max=0.9 | 53.5% | 23% | **BASELINE** |
| V2 | σ_max=1.0 + bias + scale | 39% | 33% | FAILED |
| V3 | σ_max=1.0 | 43% | 1% | FAILED |

**Conclusion:** Training on σ=1.0 (full mask) hurts generation. The model needs some signal to work with.

### Architecture Experiments

#### Option A: Self-Conditioning (FAILED)

Implemented self-conditioning: feeding previous iteration's predictions back as auxiliary input.

| Model | Config | Syntax Valid | Has = | Has Ops | Status |
|-------|--------|--------------|-------|---------|--------|
| V1 (no self-cond) | baseline | 38.5% | 20% | 84.5% | BASELINE |
| V4 (with self-cond) | use_self_cond=True | 25.0% | 16% | 84.5% | **FAILED** |
| V4 (without self-cond) | use_self_cond=False at inference | 19.5% | 13% | 84.5% | FAILED |

**Why it failed:** The model became dependent on the self-conditioning signal during training, and when starting from scratch (no previous prediction), it struggled more than V1. This is similar to the σ=1.0 problem - no signal leads to poor generation.

#### Option B: AR Warmstart (SUCCESS!)

Implemented hybrid generation: generate first N tokens autoregressively, then use MaskGIT for the rest.

| Model | Config | Syntax Valid | Has = | Has Ops | Repair | Status |
|-------|--------|--------------|-------|---------|--------|--------|
| V1 MaskGIT | baseline | 38.5% | 20% | 84.5% | 83.8% | - |
| V1 Hybrid AR3 | num_ar_tokens=3 | **57.0%** | 10% | 68.5% | 84.5% | **BEST** |
| V1 Hybrid AR5 | num_ar_tokens=5 | 57.5% | 6% | 43.0% | 84.4% | Similar |

**Why it works:** The AR warmstart provides structural anchors (BOS + first few tokens) that the bidirectional MaskGIT can build around. This addresses the "cold start" problem without requiring training changes.

**AR3 vs AR5:** AR3 is preferred because it maintains better coverage on "has equals" and "has ops" metrics while achieving similar syntax validity.

---

## Code Changes Made

### 1. data.py
Added `high_noise_bias` parameter for optional high-noise sampling:
```python
def __init__(self, ..., high_noise_bias: float = 0.0):
    # 30% of samples can be biased toward σ ∈ [0.8, sigma_max]
```

### 2. train_universal.py
- Added `--high_noise_bias` CLI argument
- Added `--use_self_cond` CLI argument for self-conditioning training
- Added self-conditioning logic with 50% dropout in training loop

### 3. model.py
- Added `use_self_cond` config flag
- Added `wte_prev` embedding layer for previous predictions (self-conditioning)
- Added `causal` parameter to `BidirectionalSelfAttention.forward()` with triangular mask
- Modified `Block.forward()` and `UniversalDenoiser.forward()` to support causal mode
- Added `prev_logits` parameter for self-conditioning

### 4. inference.py
- Added `generate_maskgit_selfcond()` for self-conditioning generation
- Added `generate_ar_tokens()` for autoregressive token generation
- Added `generate_hybrid()` for AR warmstart + MaskGIT generation
- Added batch versions: `generate_maskgit_selfcond_batch()`, `generate_hybrid_batch()`

### 5. benchmark.py
- Added `--use_self_cond` flag for self-conditioning benchmarks
- Added `--use_hybrid` and `--num_ar_tokens` flags for hybrid generation benchmarks
- Updated `benchmark_generation()` and `run_full_benchmark()` to support all modes

**Note:** Inference hacks (anchor seeding, priority unmasking) don't help current models.

---

## Checkpoints

| Path | Description | Generation | Recommendation |
|------|-------------|------------|----------------|
| `checkpoints/best.pt` | V1 baseline | 38.5% (57% w/ hybrid) | **USE WITH --use_hybrid** |
| `checkpoints_v4/best.pt` | V4 self-cond | 25% | Don't use |
| `checkpoints_v2/best.pt` | V2 scaled | 39% | Don't use |
| `checkpoints_v3/best.pt` | V3 minimal | 43% | Don't use |

---

## To Resume

**V1 + Hybrid AR3 is the best configuration (57% syntax valid).** Still 28pp short of 85% target.

Options to close the gap:

1. **Option C (Coarse→Fine):** Train separate skeleton generator - may provide better structural guidance
2. **Hybrid-aware training:** Retrain model with AR-prefix augmentation to improve hybrid performance
3. **Larger model:** Current model is only 438K params - may need more capacity

Benchmark with hybrid:
```bash
python benchmark.py --checkpoint checkpoints/best.pt --num_samples 200 --use_hybrid --num_ar_tokens 3
```

---

## Experiment Log

### V2 Results (Failed)
- Training: 50K samples, 4000 iters, σ_max=1.0, high_noise_bias=0.3
- Result: Generation dropped from 51% → 39%
- Analysis: Model produces more '=' signs but syntax is broken at all σ levels
- Conclusion: Scaling up + high_noise_bias together broke the model

### Next: V3
- Training: 20K samples, 2000 iters, σ_max=1.0, no high_noise_bias
- Hypothesis: Minimal change will work better

---

## Success Criteria
- Generation ≥ 85% syntax valid
- Repair ≥ 85% (preserve current performance)
- Editing = 100% (preserve current performance)
