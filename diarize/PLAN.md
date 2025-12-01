# Diarization SOTA: Boundary Refinement Sprint

## Hypothesis

**Boundary jitter is a significant portion of remaining DER in SOTA systems.** If we add a targeted boundary refinement stage to DiariZen (current open-source SOTA at 9.1% DER on VoxConverse, no collar), we can beat SOTA without reinventing the core diarization pipeline.

## Constraints

- **Timeline**: 2 weeks dedicated
- **Compute**: Lambda A100/H100, ~$1000 budget (efficient usage)
- **Baseline**: Start fresh with DiariZen
- **Risk tolerance**: Fail fast is acceptable; want to know quickly if this won't work

## Success Criteria

| Level | Target | Benchmark |
|-------|--------|-----------|
| **Home run** | Beat 14.49% DER | DIHARD III |
| **Strong win** | Beat 9.1% DER | VoxConverse |
| **Win** | Statistically significant DER improvement | Either benchmark |
| **Partial** | Measurable boundary precision improvement | Custom metric |
| **Fail** | No improvement after honest effort | - |

---

## Phase 1: Validation (Days 1-3)

**Goal:** Confirm our hypothesis before heavy implementation.

### Day 1: Setup & Paper Reading

**Morning:**
- [ ] Read DiariZen paper (Han et al., 2025) - understand architecture
- [ ] Read Flow-TSVAD paper (Chen et al., 2024) - closest prior art
- [ ] Check if code is open source for both

**Afternoon:**
- [ ] Clone DiariZen repo
- [ ] Set up environment (dependencies, WavLM, etc.)
- [ ] Download DIHARD III and VoxConverse datasets
- [ ] Verify we can run inference on a test file

**Deliverable:** DiariZen running locally, papers understood.

### Day 2: Baseline Evaluation

**Morning:**
- [ ] Run DiariZen on VoxConverse test set
- [ ] Run DiariZen on DIHARD III dev set
- [ ] Compare to published numbers (5.2% VoxConverse, expect ~15-17% DIHARD)

**Afternoon:**
- [ ] If numbers don't match: debug, check evaluation protocol
- [ ] Set up proper evaluation scripts (dscore or equivalent)
- [ ] Document baseline numbers

**Deliverable:** Verified baseline DER numbers.

**DECISION GATE 1:** Do we match published DiariZen performance (within 1%)?
- YES → Continue
- NO → Debug or reconsider baseline choice

### Day 3: Error Analysis

**Morning:**
- [ ] Break down DER into components: Miss, False Alarm, Speaker Confusion
- [ ] Identify boundary errors specifically:
  - Extract all speaker transitions from ground truth
  - Compare to predicted transitions
  - Measure boundary offset distribution (histogram)
- [ ] Calculate: What % of total error is boundary-related?

**Afternoon:**
- [ ] Visualize error distribution
- [ ] Document findings
- [ ] Identify specific failure cases (listen to examples)

**Deliverable:** Error analysis document with boundary error quantification.

**DECISION GATE 2:** Is boundary jitter >25% of total error?
- YES (>25%) → Proceed with refinement approach
- MAYBE (15-25%) → Proceed with caution, lower expectations
- NO (<15%) → Pivot or abort - boundary refinement won't help much

---

## Phase 2: Implementation (Days 4-7)

**Goal:** Build 3-5 boundary refinement variants that we can train and compare.

### Day 4: Architecture Design & Data Prep

**Morning:**
- [ ] Design boundary refinement interface:
  - Input: Audio window (4s around boundary), speaker embeddings (left & right)
  - Output: Refined boundary position (regression) or probability curve
- [ ] Review DiffSED architecture for coordinate regression approach
- [ ] Review Flow-TSVAD for generative refinement approach

**Afternoon:**
- [ ] Create training data pipeline:
  - Extract boundary windows from DIHARD train
  - Create synthetic boundaries (add noise to ground truth)
  - Format: (audio_window, speaker_left_embed, speaker_right_embed, true_boundary_offset)
- [ ] Set up Lambda instance, verify GPU access

**Deliverable:** Data pipeline ready, architecture designs documented.

### Day 5-6: Implement Variants (Parallel)

Implement 4 variants (can parallelize with Claude Code):

**Variant 1: MLP Baseline**
- Concatenate WavLM features around boundary + speaker embeddings
- Simple MLP predicts boundary offset
- Purpose: Sanity check - if this doesn't work, nothing will

**Variant 2: Transformer Refinement**
- Bidirectional transformer over boundary window
- Cross-attention to speaker embeddings (left and right)
- Predict boundary offset
- Purpose: Standard attention-based approach

**Variant 3: Diffusion Boundary (DiffSED-style)**
- Treat boundary position as coordinate to denoise
- Condition on audio features + speaker embeddings
- Multiple denoising steps
- Purpose: Our novel contribution

**Variant 4: Contrastive Boundary**
- Binary classifier: "Is this the correct boundary position?"
- Train with positive (true boundary) and negative (shifted boundary) pairs
- At inference: slide window, find maximum
- Purpose: Alternative framing that might work better

**Deliverable:** 4 trainable models, verified to run forward pass.

### Day 7: Training Pipeline

**Morning:**
- [ ] Set up training loop (PyTorch Lightning or similar)
- [ ] Implement evaluation metrics:
  - Boundary offset MAE/RMSE
  - Boundary accuracy at different tolerances (±50ms, ±100ms, ±200ms)
- [ ] Configure logging (wandb or tensorboard)

**Afternoon:**
- [ ] Quick sanity training runs on small data subset
- [ ] Verify all variants train without exploding gradients
- [ ] Estimate training time per variant

**Deliverable:** Training pipeline ready for full runs.

**DECISION GATE 3:** Do variants train stably? Does MLP baseline show any learning signal?
- YES → Proceed to full training
- NO → Debug architectures, simplify if needed

---

## Phase 3: Training & Evaluation (Days 8-11)

**Goal:** Train all variants, identify winner, evaluate on benchmarks.

### Day 8-9: Full Training

- [ ] Launch training for all 4 variants on Lambda
- [ ] Train on: DIHARD train + AMI train + synthetic boundaries
- [ ] Training time estimate: 10-20 hours per variant
- [ ] Monitor training curves, check for issues

**Compute budget:** ~100 GPU-hours for training phase (~$110)

### Day 10: Evaluation Round 1

**Morning:**
- [ ] Evaluate all variants on DIHARD dev (boundary metrics)
- [ ] Rank variants by boundary precision improvement
- [ ] Select top 2 variants for DER evaluation

**Afternoon:**
- [ ] Integrate best variant into DiariZen pipeline:
  - DiariZen outputs → Extract boundaries → Refine boundaries → Final output
- [ ] Evaluate end-to-end DER on DIHARD dev
- [ ] Compare to baseline DiariZen

**Deliverable:** Comparison table: baseline vs refined for each variant.

**DECISION GATE 4:** Does ANY variant improve boundary precision by >10%?
- YES → Continue to iteration
- MARGINAL (5-10%) → Continue but lower expectations
- NO (<5%) → Our approach isn't working. Consider:
  - Different architectures
  - More data
  - Or accept failure and document learnings

### Day 11: Deep Analysis

- [ ] Error analysis on best variant:
  - Where does it help?
  - Where does it fail?
  - Are there systematic patterns?
- [ ] Compare to DiariZen errors - did we fix what we intended?
- [ ] Identify potential improvements for iteration

---

## Phase 4: Iteration & Final (Days 12-14)

**Goal:** Optimize best variant, final evaluation on test sets.

### Day 12: Iteration

Based on Day 11 analysis:
- [ ] Architecture tweaks (layer count, attention heads, etc.)
- [ ] Training tweaks (learning rate, augmentation, longer training)
- [ ] Potentially combine approaches (ensemble?)
- [ ] Retrain best configuration

### Day 13: Final Evaluation

- [ ] Run best model on:
  - DIHARD III test set (final numbers)
  - VoxConverse test set (final numbers)
- [ ] Statistical significance testing
- [ ] Prepare comparison table vs published SOTA

### Day 14: Documentation & Decision

- [ ] Document full results
- [ ] Write up methodology
- [ ] Decision: Did we beat SOTA?
  - YES → Consider paper submission, integrate into diarize.io
  - NO but improved → Document learnings, decide on next steps
  - NO improvement → Document negative result, consider pivot

---

## Decision Gates Summary

| Gate | Day | Question | Pass Criteria | Fail Action |
|------|-----|----------|---------------|-------------|
| 1 | 2 | Match DiariZen baseline? | Within 1% of published | Debug or change baseline |
| 2 | 3 | Boundary jitter >25% of error? | Yes | Pivot or abort |
| 3 | 7 | Variants train stably? | MLP shows learning signal | Simplify architectures |
| 4 | 10 | Any variant improves boundaries >10%? | Yes | Consider abort |

---

## Key Papers to Read (Day 1)

1. **DiariZen** (Han et al., ICASSP 2025)
   - Architecture, training details, evaluation protocol
   - https://arxiv.org/abs/... (find exact link)

2. **Flow-TSVAD** (Chen et al., 2024)
   - Generative refinement approach
   - Closest prior art to our diffusion idea

3. **DiffSED** (Bhosale et al., AAAI 2023)
   - Diffusion for audio boundary detection
   - Architecture reference for Variant 3

4. **TS-VAD** (Microsoft, various)
   - Speaker-conditioned refinement baseline
   - Standard approach we're building on

---

## Compute Budget Estimate

| Phase | GPU Hours | Cost |
|-------|-----------|------|
| Setup & baseline | 5 | $6 |
| Data prep | 10 | $11 |
| Training (4 variants × 20h) | 80 | $88 |
| Iteration training | 40 | $44 |
| Evaluation runs | 10 | $11 |
| **Buffer (50%)** | 70 | $77 |
| **Total** | ~215 | ~$240 |

Well under $1000 budget. Room for more experimentation if needed.

---

## Risk Factors

1. **DiariZen harder to run than expected** - Mitigate: Have Pyannote as backup baseline
2. **Boundary jitter is NOT the main error** - Mitigate: Decision Gate 2 catches this early
3. **Refinement doesn't help** - Mitigate: 4 variants increase odds, MLP sanity check
4. **Training instabilities** - Mitigate: Start simple (MLP), proven architectures
5. **Evaluation protocol mismatch** - Mitigate: Use official eval scripts (dscore)

---

## What Success Looks Like

**Best case (Day 14):**
- Beat DIHARD III SOTA (14.49% → 13.x% DER)
- Clear evidence boundary refinement is the cause
- Publishable result + product improvement

**Good case:**
- Measurable improvement (0.5-1% DER reduction)
- Validated approach, clear path to further improvement
- Worth continuing development

**Acceptable case:**
- Boundary precision improved but DER unchanged
- Learned why (e.g., boundary errors were already low, or improvements don't translate to DER)
- Informed decision about next steps

**Fail case:**
- No improvement despite honest effort
- Documented learnings about why
- Clear signal to pursue different approach

---

## Questions Resolved

- ✅ GPU access: Lambda A100/H100, ~$1000 budget
- ✅ Baseline: DiariZen (start fresh)
- ✅ Timeline: 2 weeks dedicated
- ✅ Risk tolerance: Fail fast acceptable
- ✅ Data: Use public datasets (DIHARD, AMI, VoxConverse)

## Open Questions

- [x] Is DiariZen code publicly available and runnable? **YES** - GitHub + HuggingFace models
- [x] Is Flow-TSVAD code available? **NO** - paper only, implement from scratch
- [ ] Exact DIHARD evaluation protocol (collar, overlap handling)

## Verified Resources

**DiariZen:**
- Repo: https://github.com/BUTSpeechFIT/DiariZen
- Paper: https://arxiv.org/abs/2409.09408
- Models: `BUT-FIT/diarizen-wavlm-large-s80-md` on HuggingFace
- License: MIT (code), research-only (models)
- Load: `DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md")`

**Flow-TSVAD:**
- Paper: https://arxiv.org/abs/2409.04859
- Code: NOT RELEASED - implement from paper
- Key insight: Maps binary labels → continuous latent → flow matching refinement
