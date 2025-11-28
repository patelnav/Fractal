# Fixing Generation in Phase 31

**Problem:** Generation from full-mask achieves only 46% syntax validity (target: 85%)

**Root Cause:** Training distribution mismatch (σ ∈ [0.1, 0.9], never sees σ = 1.0). Model has no signal to bootstrap from when all tokens are MASK.

**Key Insight:** Model capacity is sufficient (repair: 85%, editing: 100%). This is a cold-start problem, not a scaling problem.

---

## Two Paths Forward

### Path 1: Fix the Training Distribution (Direct Fix)

**Retrain with σ ∈ [0.1, 1.0] + high-noise bias**

This directly addresses the root cause. The model was never trained on σ = 1.0 (full mask), so it's out-of-distribution at generation time.

```python
# Current (broken)
sigma = torch.rand(batch_size) * 0.8 + 0.1  # σ ∈ [0.1, 0.9]

# Fixed
sigma = torch.rand(batch_size) * 0.9 + 0.1  # σ ∈ [0.1, 1.0]
# Plus: 30% of batches force σ > 0.8
if random.random() < 0.3:
    sigma = torch.rand(batch_size) * 0.2 + 0.8  # σ ∈ [0.8, 1.0]
```

| Aspect | Value |
|--------|-------|
| Effort | 4 hours (retrain existing architecture) |
| Expected gain | +20-30% (46% → 66-76%) |
| Risk | Low - same architecture, just different data distribution |
| Preserves | Repair and editing performance (they use lower σ) |

**Why this might be all you need:** The model already works for repair (σ ≈ 0.2) and editing (σ ≈ 0.5). It just needs to see the high-noise regime during training.

### Path 2: Inference-Time Workarounds (Hacks)

If you don't want to retrain, stack inference-time fixes that compensate for the missing training signal. These are workarounds that reduce the effective "coldness" of the cold-start.

**Tradeoff:**
- Path 1 is cleaner but costs 4 hours of training
- Path 2 is faster to test but adds inference complexity

**Recommendation:** Try Path 2 experiments first (they take <2 hours total). If they don't reach 85%, do Path 1. The fixes may be complementary - Path 1 + Path 2 together could exceed either alone.

---

## Three Solution Clusters

### Cluster A: "Give the Model a Hint"
Reduce the cold-start problem by providing partial structure.

### Cluster B: "Better Unmasking Order"
The order of token decisions matters. Lock in skeleton before details.

### Cluster C: "Different Diffusion Formulation"
Random mask → denoise may not be the optimal path.

---

## Prioritized Experiments

### Tier 1: Inference-Only (No Retraining)

#### 1. Structural Anchor Seeding
**Effort:** 5 min | **Expected Gain:** +15-20%

Always fix BOS, `=`, and EOS before generation starts. Model fills left/right of `=` separately.

```python
def generate_with_anchor(model, length):
    x = torch.full((1, length), MASK_ID)
    x[0, 0] = BOS_ID
    x[0, -1] = EOS_ID
    eq_pos = length * 2 // 3  # heuristic
    x[0, eq_pos] = EQ_ID
    # Run MaskGIT, never unmask anchor positions
```

#### 2. Priority Unmasking (Structure First)
**Effort:** 15 min | **Expected Gain:** +10-15%

Unmask operators and parentheses before digits. Ensures syntactic skeleton forms first.

```python
def get_unmask_priority(logits, x_current):
    confidence = logits.softmax(-1).max(-1).values
    is_structural = logits.argmax(-1).isin([LPAREN, RPAREN, PLUS, MULT, EQ])
    priority = confidence + 0.5 * is_structural  # boost structural tokens
    return priority
```

#### 3. Energy-Guided Rejection
**Effort:** 30 min | **Expected Gain:** +5-10%

Use existing energy head to reject bad intermediate states during MaskGIT.

```python
for step in range(n_steps):
    candidates = [sample() for _ in range(k)]
    energies = [energy_head(c) for c in candidates]
    x = candidates[argmin(energies)]
```

---

### Tier 2: Minor Code Changes

#### 4. Self-Conditioning
**Effort:** 1 hr | **Expected Gain:** +10-20%

Feed previous iteration's prediction as auxiliary input. Proven on Imagen, SoundStorm.

```python
# In forward pass
if self.prev_logits is not None:
    x_prev_estimate = self.prev_logits.argmax(-1)
    hidden = self.embed(x_noisy) + 0.5 * self.embed(x_prev_estimate) + self.noise_embed(σ)
else:
    hidden = self.embed(x_noisy) + self.noise_embed(σ)
```

#### 5. AR Warmstart (3 tokens)
**Effort:** 2 hr | **Expected Gain:** +15-25%

Generate first 3 tokens autoregressively, then switch to bidirectional fill.

```python
def generate_hybrid(model, length):
    # Phase 1: AR generates skeleton
    x = [BOS_ID]
    for _ in range(3):
        logits = causal_model(x)
        x.append(sample(logits[-1]))

    # Phase 2: Pad with MASK, run bidirectional
    x = x + [MASK_ID] * (length - len(x) - 1) + [EOS_ID]
    return maskgit_refine(model, x)
```

---

### Tier 3: Requires Retraining

#### 6. Consistency Distillation
**Effort:** 1 day | **Expected Gain:** +20-30%

Train model to predict x₀ directly from any x_t. Eliminates multi-step error accumulation.

```python
# Consistency loss: predictions from adjacent timesteps should match
loss = ||f(x_t, t) - f(x_{t+1}, t+1)||
```

#### 7. Discrete Flow Matching
**Effort:** 2 days | **Expected Gain:** +15-25%

Replace random corruption with optimal transport paths from clean → mask.

#### 8. Retrain with σ ∈ [0.1, 1.0] (Path 1 - Direct Fix)
**Effort:** 4 hrs | **Expected Gain:** +20-30%

See "Two Paths Forward" above. This is the direct fix to the root cause. Bias 30% of batches toward σ > 0.8.

**Note:** This can be combined with Path 2 inference hacks for potentially additive gains.

---

## Recommended Execution Order

```
┌─────────────────────────────────────────────────────────────────┐
│  DECISION: Do you want the "proper fix" or "quick experiments"? │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
    ┌───────────────┐                   ┌───────────────┐
    │   Path 1:     │                   │   Path 2:     │
    │   Retrain     │                   │   Inference   │
    │   (4 hours)   │                   │   Hacks       │
    └───────┬───────┘                   └───────┬───────┘
            │                                   │
            ▼                                   ▼
    σ ∈ [0.1, 1.0]                      Stack experiments:
    + 30% high-noise bias               1. Anchor seeding (5 min)
            │                           2. Priority unmasking (15 min)
            │                           3. Energy rejection (30 min)
            ▼                           4. Self-conditioning (1 hr)
    Expected: 66-76%                            │
            │                                   ▼
            │                           Expected: 60-75%
            │                                   │
            └───────────┬───────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Still below 85%?     │
            │  Combine Path 1 + 2   │
            │  Expected: 80-90%     │
            └───────────────────────┘
```

**Fastest to 85%:** Path 1 (retrain) alone may be sufficient. It's 4 hours of training but addresses root cause directly.

**Lowest risk:** Path 2 first (2 hours of inference experiments), then Path 1 if needed.

**Maximum quality:** Both paths combined - fixes the training distribution AND adds inference-time structure.

---

## Success Criteria

### Path 1 (Retrain)

| Experiment | Minimum | Target |
|------------|---------|--------|
| Retrain with σ_max=1.0 | 65% | 76% |
| + high-noise bias (30%) | 70% | 80% |
| + Path 2 inference hacks | 80% | **90%** |

### Path 2 (Inference Hacks Only)

| Experiment | Minimum | Target |
|------------|---------|--------|
| Anchor seeding alone | 55% | 65% |
| + Priority unmasking | 60% | 70% |
| + Self-conditioning | 70% | 80% |
| + AR warmstart | 80% | 88% |

If either path reaches 85%+, the unified denoising thesis is validated. Path 1 is the "proper" solution; Path 2 proves you can compensate at inference time.

---

## Dead Ends (Don't Try)

| Idea | Why Not |
|------|---------|
| More layers / bigger model | Capacity isn't the bottleneck (repair/edit work) |
| Classifier-free guidance | No condition to guide against |
| Contrastive decoding | Needs two models |
| Pure causal parallel | Phase 28-29 proved this fails |
| Template retrieval | Too task-specific |

---

## Key Insight

The generation problem is **initialization**, not **capacity**. Every proposed fix either:
1. Provides better initialization (anchors, AR warmstart)
2. Makes smarter decisions about what to commit first (priority unmasking)
3. Gives the model memory of its previous guesses (self-conditioning)

None require scaling the model.
