# Next Steps - Post Phase 3

**Status:** Phase 3 COMPLETE | **Result:** Hypothesis Falsified
**Date:** 2024-12-02

---

## Phase 3 Results Summary

### Training Results (Synthetic Validation)

| Model | Params | Best val_mae | Epoch |
|-------|--------|--------------|-------|
| MLP | 822K | **2.2ms** | 15 |
| Transformer | 1.4M | 2.9ms | 1 |
| Contrastive | 361K | 21.8ms | 0 |

### Real-World Evaluation (on DiariZen Predictions)

| Model | Boundary Improvement | Status |
|-------|---------------------|--------|
| Transformer | **0.0%** | No effect |
| MLP | **-1.2%** | Made it worse |

### Key Discovery

| Metric | Day 3 Estimate | Actual (Phase 3) |
|--------|----------------|------------------|
| DiariZen boundary error | 923ms | **188.8ms** |
| Headroom for improvement | ~800ms | ~100ms |

**The original 923ms estimate was incorrect.** DiariZen's actual boundary error is 188.8ms mean / 88ms median - already quite good.

---

## Decision Gate 4: FAIL

From PLAN.md:
> "Does ANY variant improve boundary precision by >10%?"

**Result:** No. Best variant (Transformer) showed 0.0% improvement.

**Root Cause:** Distribution mismatch - models trained on synthetic Gaussian noise (σ=300ms) don't generalize to real DiariZen errors (188.8ms mean, non-Gaussian).

---

## Why the Hypothesis Was Wrong

**Original Hypothesis:**
> "Boundary jitter (~923ms) is ~50% of remaining DER. Targeted refinement can beat SOTA."

**Reality:**
1. DiariZen's boundary error is 188.8ms, not 923ms
2. 55% of boundaries are already within 100ms
3. 80% of boundaries are already within 200ms
4. There's no significant boundary problem to fix

The 923ms figure from Day 3 likely measured something different (all segment boundaries vs speaker-change transitions).

---

## Options Going Forward

### Option A: Different Training Approach
If still pursuing boundary refinement:
- Train on actual DiariZen predictions → GT (not synthetic noise)
- Analyze DiariZen error distribution first (bimodal? clustered?)
- Use real speaker embeddings (not dummy hashes)
- Add confidence-gated refinement (skip low-confidence predictions)

**Estimate:** ~5 decision points, ~$20 compute, 1-2 days

### Option B: Different Error Target
If boundary isn't the problem, target what is:
- Overlap handling (multi-speaker regions)
- Short segment recovery
- VAD boundary refinement (speech/silence boundaries)
- Speaker confusion in similar-voice scenarios

**Estimate:** Requires new error analysis first

### Option C: Accept Current Performance
- DiariZen achieves 4.52% DER on VoxConverse dev
- This may be near practical ceiling for this dataset
- Focus resources elsewhere

---

## Artifacts Saved

```
diarize/
├── checkpoints/
│   ├── mlp_best.ckpt (9.4MB)
│   ├── transformer_best.ckpt (41MB)
│   └── contrastive_best.ckpt (8.3MB)
├── logs/
│   ├── mlp/train_mlp.log + TensorBoard
│   └── transformer/train_transformer.log + TensorBoard
├── results/voxconverse_dev/
│   └── 216 RTTM files (DiariZen predictions)
└── EXPERIMENT_RESULTS.md (full documentation)
```

---

## Budget Status

| Item | Cost |
|------|------|
| Phase 1-2 (setup, baseline) | $0.72 |
| Phase 3 (training, eval) | ~$20 |
| **Total spent** | **~$21** |
| **Budget remaining** | **~$979** |

---

## References

- **LOG.md:** Full research log (Phases 1-3)
- **PLAN.md:** Original plan with decision gates
- **EXPERIMENT_RESULTS.md:** Detailed Phase 3 results
