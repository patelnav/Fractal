# Boundary Refinement Experiment Results

**Date**: December 2, 2024
**Dataset**: VoxConverse dev (216 files)
**Baseline System**: DiariZen (BUT-FIT/diarizen-wavlm-large-s80-md)

---

## Executive Summary

**Key Finding**: Models trained on synthetic noise (Gaussian perturbations) **do not generalize** to real DiariZen prediction errors. Both MLP and Transformer models showed negative or zero improvement on real data despite excellent synthetic validation metrics.

| Model | Synthetic val_mae | Real Improvement | Status |
|-------|-------------------|------------------|--------|
| MLP | **2.2ms** | **-1.2%** (worse) | FAILED |
| Transformer | **2.9ms** | **0.0%** (neutral) | NO IMPACT |
| Contrastive | 21.8ms | Not tested | - |

---

## 1. Training Results

### Training Setup
- **Hardware**: 3x Lambda A100 (40GB SXM4) instances
- **Cost**: ~$20 total for ~6 hours training
- **Framework**: PyTorch Lightning + WavLM features
- **Data**: VoxConverse dev set (216 audio files, ~8k real boundaries, 32k synthetic)

### Model Architectures

**MLP Baseline**
- Hidden dims: [512, 256, 128]
- Input: WavLM pooled features (768-dim) + speaker embeddings (256-dim)
- 20 epochs, early stopping patience=5

**Transformer**
- 4 layers, 8 heads, d_model=256, FFN=1024
- Attention over WavLM frame features
- Early stopped at epoch 6

**Contrastive**
- Contrastive learning with triplet loss
- Learned boundary vs non-boundary embeddings
- Early stopped at epoch 5

### Validation Metrics (Synthetic Test Data)

| Model | Best val_mae | Epoch | Checkpoint Size |
|-------|--------------|-------|-----------------|
| **MLP** | **2.2ms** | 15 | 9.8MB |
| Transformer | 2.9ms | 1 | 42MB |
| Contrastive | 21.8ms | 0 | 8.6MB |

---

## 2. Real-World Evaluation

### DiariZen Baseline Boundary Error

**Critical Finding**: DiariZen's actual boundary error is much better than previously reported.

| Metric | Value |
|--------|-------|
| Mean error | **188.8ms** |
| Median error | 88.0ms |
| Std error | 322.8ms |
| @50ms accuracy | 29.3% |
| @100ms accuracy | 55.3% |
| @200ms accuracy | 79.7% |
| @500ms accuracy | 91.7% |

**Note**: Previous estimates of ~923ms were likely from a different error calculation or dataset subset.

### MLP Model on Real DiariZen Predictions

```
Total boundaries evaluated: 4237

BEFORE refinement:
  Mean error:   188.8 ms
  Median error: 88.0 ms

AFTER refinement:
  Mean error:   191.1 ms  (+2.3ms WORSE)
  Median error: 90.9 ms   (+2.9ms WORSE)

Improvement: -1.2% (negative = degradation)
Boundaries improved: 1995/4237 (47.1%)

Accuracy changes:
  @50ms: 29.3% -> 28.7% (-0.6%)
  @100ms: 55.3% -> 53.2% (-2.2%)
  @200ms: 79.7% -> 79.0% (-0.7%)
  @500ms: 91.7% -> 91.7% (+0.0%)
```

### Transformer Model on Real DiariZen Predictions

```
Total boundaries evaluated: 4237

BEFORE refinement:
  Mean error:   188.8 ms
  Median error: 88.0 ms

AFTER refinement:
  Mean error:   188.8 ms  (UNCHANGED)
  Median error: 89.9 ms   (+1.9ms slightly worse)

Improvement: 0.0% (neutral)
Boundaries improved: 2148/4237 (50.7%)

Accuracy changes:
  @50ms: 29.3% -> 30.0% (+0.8%)  <- slight improvement!
  @100ms: 55.3% -> 55.3% (+0.0%)
  @200ms: 79.7% -> 79.7% (+0.0%)
  @500ms: 91.7% -> 91.7% (+0.0%)
```

**Analysis**: Transformer is neutral overall but shows slight improvement at @50ms threshold. The model helps ~51% of boundaries but the improvements/degradations cancel out. This is better than MLP (-1.2%) but still not useful.

---

## 3. Key Learnings

### What Went Wrong

1. **Distribution Mismatch**: Training data used Gaussian noise (std=0.3s = 300ms), but real DiariZen errors have:
   - Median 88ms (much smaller than 300ms noise)
   - Heavy tail (322ms std) - some very large errors
   - Likely non-Gaussian, bimodal or clustered patterns

2. **Model Overfitting to Synthetic Noise**: The 2.2ms validation MAE means the model learned to predict noise center perfectly - but real boundaries aren't centered at the predicted position plus Gaussian noise.

3. **Speaker Embeddings Were Dummy**: Training used deterministic dummy embeddings (hash of speaker ID), not real speaker representations. This likely hurt the model's ability to distinguish speaker-specific boundary patterns.

4. **Boundary Matching Tolerance**: Used 2.0s max distance for matching - this may have included spurious matches.

### What the Results Tell Us

1. **DiariZen is Already Good**: 188.8ms mean error with 55% within 100ms is competitive. The headroom for improvement is smaller than expected.

2. **Synthetic Training Doesn't Generalize**: Models trained on artificially perturbed boundaries don't learn to correct real system errors.

3. **~47% of boundaries still improved**: While overall metrics degraded, nearly half of individual boundaries got better. The model isn't completely wrong - it's just making some boundaries worse by more than it improves others.

---

## 4. Recommendations for Future Work

### Data Strategy
1. **Train on real DiariZen errors**: Use DiariZen predictions as input, ground truth as targets
2. **Analyze error distribution**: Characterize real DiariZen error patterns (clustering, bimodality, etc.)
3. **Add real speaker embeddings**: Use WeSpeaker or similar for actual speaker representations

### Architecture Ideas
1. **Asymmetric loss**: Penalize making boundaries worse more than failing to improve
2. **Confidence prediction**: Output confidence to skip low-confidence refinements
3. **Iterative refinement**: Multiple passes with decreasing step sizes
4. **Error-type classification**: First classify error type, then apply type-specific correction

### Evaluation
1. **Per-boundary analysis**: Which boundaries improve vs degrade?
2. **Error magnitude bucketing**: Does the model help large errors more than small ones?
3. **DER impact**: What's the actual DER change from boundary refinement?

---

## 5. Files and Artifacts

### Local Checkpoints
```
diarize/checkpoints/
├── contrastive_best.ckpt (8.6MB)
├── mlp_best.ckpt (9.8MB)
└── transformer_best.ckpt (42MB)
```

### Training Logs
```
diarize/logs/
├── mlp/
│   ├── train_mlp.log
│   └── logs/version_*/events.*
└── transformer/
    ├── train_transformer.log
    └── logs/version_*/events.*
```

### DiariZen Predictions
```
diarize/results/voxconverse_dev/
└── *.rttm (216 files)
```

---

## 6. Instance Status

| Instance | Status | Notes |
|----------|--------|-------|
| lambda | **TERMINATED** | All artifacts downloaded |
| lambda-mlp | **TERMINATED** | All artifacts downloaded |
| lambda-contrastive | **TERMINATED** | All artifacts downloaded |

---

*Document created: December 2, 2024*
*Last updated: December 2, 2024 - All evaluations complete*
