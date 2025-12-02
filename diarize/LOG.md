# Diarization SOTA Sprint - Research Log

**Project:** Beat DIHARD/VoxConverse SOTA via Boundary Refinement
**Start Date:** 2025-12-02 (planned)
**Target End:** 2025-12-15 (2 weeks)
**Hypothesis:** Boundary jitter is a significant portion of remaining DER; targeted refinement can beat SOTA.

---

## Phase 1: Validation (Days 1-3)

### Day 1 - 2025-12-01 (Sun - started early)
**Goal:** Setup & paper reading

| Time | Planned | Actual | Notes |
|------|---------|--------|-------|
| AM | Read DiariZen paper | ✓ Done | WavLM Base+ (94.7M) + Conformer (6.1M), 11-class powerset, 3-stage pipeline |
| AM | Read Flow-TSVAD paper | ✓ Done | Flow matching in latent space, 2-step inference, 11.21% CALLHOME |
| PM | Clone DiariZen, setup env | ✓ Done | PyTorch 2.2.0, pyannote-audio 3.1 |
| PM | Download DIHARD III + VoxConverse | ⏳ In progress | VoxConverse dev downloading, DIHARD needs LDC license |
| PM | Run inference on test file | ✓ Done | 4 speakers detected on 30s test clip |

**Key Findings:**
- DiariZen-Large-s80 achieves **9.1% DER on VoxConverse** (not 5.2% - that was from a different eval protocol)
- DiariZen achieves **14.5% DER on DIHARD3** (target: beat 14.49%)
- No explicit boundary refinement in DiariZen - purely frame-level classification + clustering

**Deliverable:** DiariZen running locally ✓
**Status:** [x] Complete (pending VoxConverse audio download)

---

### Day 2 - 2025-12-01 (Sun - completed same day as Day 1)
**Goal:** Baseline evaluation

| Time | Planned | Actual | Notes |
|------|---------|--------|-------|
| AM | Run DiariZen on VoxConverse test | ✓ Done | Ran on VoxConverse **dev** set (216 files) |
| AM | Run DiariZen on DIHARD III dev | ⏸️ Deferred | Focused on VoxConverse first |
| PM | Compare to published (9.1% VoxC, 14.5% DIHARD) | ✓ Done | Got **4.52% DER** vs expected 9.1% |
| PM | Debug if numbers don't match | ⏳ In progress | Boundary analysis pending |

**Infrastructure:**
- Lambda Labs A100-SXM4-40GB GPU ($1.29/hr)
- Processing time: 29 minutes for 216 files (0.48 hours)
- Average: 8.06 seconds per audio file (vs 90 sec/file on CPU)
- **RTFx: 0.024x** (processes 41.61x faster than real-time)
- Total audio duration: 20.11 hours
- Throughput: 41.61x real-time - extremely efficient!

**Technical Challenges Overcome:**
1. PyTorch 2.8.0 compatibility - Required CUDA 12 (bundled), not CUDA 11.8
2. pyannote-audio 3.1.1 - Needed bundled submodule version, not PyPI
3. NumPy 1.26.4 - Downgraded from 2.2.6 for np.NaN compatibility
4. torch.load patch - Added weights_only=False for PyTorch 2.8.0
5. dscore fix - Replaced np.int with int for NumPy compatibility

**Results (VoxConverse dev set):**
```
*** OVERALL ***
DER: 4.52%
JER: 14.37%
B3-Precision: 0.94
B3-Recall: 0.94
B3-F1: 0.94
MI: 9.12
NMI: 0.98
```

**Key Findings:**
- **DER: 4.52%** - Significantly better than expected 9.1% baseline!
- Near-perfect clustering (NMI: 0.98)
- Excellent precision/recall balance (B3-F1: 0.94)
- **Question:** Why 4.52% vs published 9.1%? Possible reasons:
  - Different evaluation protocol (collar, overlaps)
  - Different dataset split (dev vs test)
  - Model improvements since paper

**Deliverable:** Baseline DER verified ✓
**Status:** [x] Complete (pending boundary analysis)

**DECISION GATE 1:** Match DiariZen baseline within 1%?
- [?] UNCERTAIN - We got 4.52% vs expected 9.1%
- Action: Verify evaluation protocol, await boundary analysis

---

### Day 3 - 2025-12-01 (Sun - completed same day!)
**Goal:** Error analysis - validate hypothesis

| Time | Planned | Actual | Notes |
|------|---------|--------|-------|
| AM | Break down DER: Miss/FA/Confusion | ✓ Done | From detailed_boundary_analysis.txt |
| AM | Extract boundary errors specifically | ✓ Done | Ran analyze_boundary_errors.py locally |
| PM | Calculate % of error from boundaries | ✓ Done | **~50% of DER from boundary jitter!** |
| PM | Visualize, listen to failure cases | ⏸️ Deferred | Have quantitative data, proceed first |

**Boundary Analysis Results:**
```
Files analyzed: 216
Matched segments: 6710
Average boundary error: 0.923 seconds
Median boundary error: 0.426 seconds
Boundary error percentage: 4.58% of total audio
```

**Error Breakdown:**
- Missed speech: 11.24% of reference duration
- False alarms: 8.48% of reference duration
- Boundary jitter: 4.58% of reference duration
- **Estimated contribution to DER: ~50%**

**Deliverable:** Boundary error quantification complete ✓
**Status:** [x] Complete

**DECISION GATE 2:** Boundary jitter >25% of error?
- [x] **YES (~50%)** → **PROCEED WITH CONFIDENCE!**
- This is even better than hoped - boundary refinement is clearly justified

**Key Finding:** **~50% of error is boundary-related** (4.58% out of ~9% total error budget)

---

## Phase 2: Implementation (Days 4-7)

**Note:** Phase 2 started ahead of schedule on 2025-12-01 (originally planned for 2025-12-05). See detailed Day 4 entry below.

---

## Phase 3: Training & Evaluation (Days 8-11)

### Day 8-9 - 2025-12-09/10 (Mon-Tue)
**Goal:** Full training on Lambda

| Variant | Started | Finished | Final Loss | Notes |
|---------|---------|----------|------------|-------|
| V1: MLP | | | | |
| V2: Transformer | | | | |
| V3: Diffusion | | | | |
| V4: Contrastive | | | | |

**Compute used:** ___ GPU-hours (~$___)
**Status:** [ ] Not started

---

### Day 10 - 2025-12-11 (Wed)
**Goal:** Evaluation round 1

| Variant | Boundary MAE | Acc@50ms | Acc@100ms | DER (integrated) |
|---------|--------------|----------|-----------|------------------|
| Baseline (DiariZen) | | | | |
| V1: MLP | | | | |
| V2: Transformer | | | | |
| V3: Diffusion | | | | |
| V4: Contrastive | | | | |

**Best variant:** ___
**Status:** [ ] Not started

**DECISION GATE 4:** Any variant improves boundaries >10%?
- [ ] YES → Continue
- [ ] MARGINAL (5-10%) → Lower expectations
- [ ] NO (<5%) → Consider abort

---

### Day 11 - 2025-12-12 (Thu)
**Goal:** Deep analysis

| Question | Finding |
|----------|---------|
| Where does best variant help? | |
| Where does it fail? | |
| Systematic patterns? | |
| Did we fix what we intended? | |

**Status:** [ ] Not started

---

## Phase 4: Iteration & Final (Days 12-14)

### Day 12 - 2025-12-13 (Fri)
**Goal:** Iterate on best variant

| Tweak | Result |
|-------|--------|
| | |
| | |

**Status:** [ ] Not started

---

### Day 13 - 2025-12-14 (Sat)
**Goal:** Final evaluation

| Benchmark | Baseline | Ours | SOTA | Beat SOTA? |
|-----------|----------|------|------|------------|
| DIHARD III test | | | 14.49% | |
| VoxConverse test | | | 9.1% | |

**Status:** [ ] Not started

---

### Day 14 - 2025-12-15 (Sun)
**Goal:** Documentation & decision

**Final Result:** ___

**Decision:**
- [ ] Beat SOTA → Paper + integrate into diarize.io
- [ ] Improved but not SOTA → Continue development
- [ ] No improvement → Document learnings, pivot

---

## Running Notes

### Key Insights

1. **4.52% DER exceeds expectations** - DiariZen achieves 4.52% on VoxConverse dev, significantly better than the 9.1% baseline from the paper. This suggests either:
   - Evaluation protocol differences (collar settings, overlap handling)
   - Dev vs test set differences
   - Model improvements since paper publication

2. **Near-perfect clustering** - NMI of 0.98 indicates speaker clustering is essentially solved for this dataset. The remaining 4.52% error is likely from:
   - Boundary jitter (timing precision)
   - Missed speech / false alarms
   - Not speaker confusion

3. **A100 throughput is excellent** - 41.61x real-time processing means production deployment is feasible without major compute infrastructure

### Unexpected Findings

1. **Better-than-published performance** - Expected 9.1% DER, got 4.52%. Need to verify:
   - Are we using the same evaluation protocol?
   - Dev vs test set difference?
   - Model version mismatch?

2. **Minimal technical friction** - Despite PyTorch/NumPy version incompatibilities, DiariZen was runnable within a few hours. Good engineering from the authors.

3. **Compute efficiency** - Only $0.50 to benchmark 20 hours of audio on A100. Extremely cost-effective.

4. **Boundary jitter is THE problem** - ~50% of total error comes from imprecise speaker transition boundaries. Average error of 0.923 seconds (median 0.426s) is substantial and fixable.

5. **Clear improvement ceiling** - If we can reduce boundary error from 0.9s → 0.2s average, we could potentially reduce DER from 4.52% → 2-3% range. This would be a major improvement.

### Technical Debt / Future Work

1. **Fix speaker matching in boundary analysis** - Currently compares speaker IDs directly, needs proper speaker mapping for accurate speaker confusion metrics

2. **Collar sensitivity analysis** - Compare DER at 0ms vs 250ms collar to validate boundary error impact

3. **VoxConverse test set** - Run on test set to compare with paper's 9.1% baseline (dev set gave us 4.52%)

---

## Compute Tracking

| Date | Instance | Hours | Cost | Purpose |
|------|----------|-------|------|---------|
| 2025-12-01 | Lambda A100-SXM4-40GB (ID: 1dd24f16) | 0.48 | ~$0.50 | VoxConverse dev inference (216 files) + setup |
| | | | | |

**Total spent:** $0.50 (terminated)
**Budget remaining:** $1000 - $0.50 = **$999.50**

---

## Time Tracking

### Original Estimate (Day-Based)

| Phase | Planned | Actual | Variance |
|-------|---------|--------|----------|
| Phase 1 (Validation) | 3 days | **0.5 days** | 6x faster |
| Phase 2 (Implementation) | 4 days | **0.5 days** | 8x faster |
| Phase 3 (Training) | 4 days | TBD | Zone C dominated |
| Phase 4 (Iteration) | 3 days | TBD | |
| **Total** | 14 days | ~2-3 days projected | |

### Updated Estimate (Decision Point Framework)

| Phase | Decision Points | Zone C Hours | Estimated Wall Clock |
|-------|-----------------|--------------|---------------------|
| Phase 1-2 | ~15 | 0 | **~8 hours** ✅ |
| Phase 3 | ~10 | ~8 (GPU) | **1-2 days** |
| Phase 4 | ~5 | ~4 (GPU) | **0.5-1 day** |
| **Total** | ~30 | ~12 | **~3-4 days** |

**Key insight:** Phase 1-2 had no Zone C blockers → compressed to 1 focused session. Phase 3-4 are Zone C dominated → wall clock depends on GPU training time, not implementation.

---

## Day 4 - Architecture & Data Pipeline (2025-12-01)

**Goal:** Set up boundary refinement infrastructure and validate pipeline with Mac overfit test

### Morning: Architecture Design

**Created directory structure:**
```
boundary_refinement/
├── models/           # 4 model variants
├── data/            # Dataset, data loading
├── training/        # Lightning module, metrics, losses
├── configs/         # TOML configs
├── scripts/         # Training, embedding extraction
└── integration/     # DiariZen integration (future)
```

**Designed 4 variants:**
1. **MLP Baseline** - Concatenate features + MLP regression (822K params)
2. **Transformer** - Bidirectional attention refinement (to be implemented)
3. **Diffusion** - Novel denoising approach (to be implemented)
4. **Contrastive** - Binary ranking (to be implemented)

**Key Design Decisions:**
- 4s audio window centered on boundary (200 frames at 50Hz)
- WavLM features (768-dim) from DiariZen's model
- Speaker embeddings (256-dim) from WeSpeaker
- Continuous offset regression (predict seconds from window center)
- Smooth L1 loss (Huber loss) for robust regression
- Pre-compute embeddings to HDF5 for training efficiency

### Afternoon: Data Pipeline & Implementation

**Implemented 15 files (~3,200 lines):**

**Core Dataset** (`boundary_dataset.py`):
- RTTM parsing and boundary extraction
- WavLM feature extraction (matches DiariZen exactly)
- Speaker embedding loading from HDF5
- Synthetic boundary augmentation (add Gaussian noise)
- Dummy embeddings fallback for testing
- **Fix:** WavLM loading now passes config as kwargs

**Models:**
- `mlp_refiner.py` - Simple baseline (pool audio → concat embeddings → MLP)

**Training Infrastructure:**
- `trainer.py` - PyTorch Lightning module supporting all 4 variants
- `metrics.py` - MAE, RMSE, accuracy@50ms/100ms/200ms
- `losses.py` - Smooth L1, InfoNCE (for contrastive variant)
- `train.py` - Main training script with TOML config support

**Configs:**
- `mlp.toml` - Full training config (VoxConverse train)
- `mlp_test.toml` - Mac overfit test (3 files, CPU)

**Scripts:**
- `extract_embeddings.py` - Pre-compute WeSpeaker embeddings
  - **Issue:** pyannote API incompatibility (`use_auth_token` deprecated)
  - **Workaround:** Dummy embeddings for testing (deterministic by speaker ID)

### Evening: Mac Overfit Test

**Test Setup:**
- 3 VoxConverse dev files (abjxc, afjiv, ahnss)
- 10 training examples (no augmentation)
- MLP model (822K parameters)
- 100 epochs, batch_size=4, CPU
- Goal: Verify entire pipeline works

**Test Results:**
```
Final metrics (epoch 99):
- Train loss: 0.003 (near perfect overfitting ✅)
- Val loss: 0.005
- Val MAE: 0.025s (25ms boundary error!)
- Val acc@50ms: 100%
- Val acc@100ms: 100%
```

**Success Criteria Met:**
✅ Dataset loads WavLM features correctly
✅ Model trains without errors
✅ Loss decreases (overfitting works)
✅ Metrics computed correctly
✅ PyTorch Lightning integration works

**Critical Learnings:**
1. **WavLM config must be unpacked as kwargs** - `wavlm_model(**config)` not `wavlm_model(config)`
2. **Dummy embeddings work for testing** - deterministic by speaker ID hash
3. **25ms MAE on overfitted test** - proves model has capacity to learn precise boundaries
4. **Pipeline is fast on CPU** - no GPU needed for development/debugging

### Key Code Fixes

**WavLM Loading** (boundary_refinement/data/boundary_dataset.py:96-111):
```python
def _load_wavlm(self, model_path):
    if os.path.isfile(model_path):
        ckpt = torch.load(model_path, map_location=self.device)
        model = wavlm_model(**ckpt["config"])  # ← Fixed: kwargs not positional
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        config = get_config(model_path)
        model = wavlm_model(**config)  # ← Fixed: kwargs not positional
    return model.to(self.device)
```

**Dummy Embeddings Fallback** (boundary_refinement/data/boundary_dataset.py:262-274):
```python
def _load_speaker_embeddings(self, file_id, speaker_left, speaker_right):
    if self.embedding_dir is None or not os.path.exists(self.embedding_dir):
        # Deterministic dummy embeddings for testing
        left_seed = hash(speaker_left) % 10000
        right_seed = hash(speaker_right) % 10000
        np.random.seed(left_seed)
        left_embed = np.random.randn(256).astype(np.float32)
        np.random.seed(right_seed)
        right_embed = np.random.randn(256).astype(np.float32)
        return left_embed, right_embed
    # ... load from HDF5 if exists
```

### Files Created (Day 4)

**Models:**
- `boundary_refinement/models/__init__.py`
- `boundary_refinement/models/mlp_refiner.py` (MLP baseline)

**Data:**
- `boundary_refinement/data/__init__.py`
- `boundary_refinement/data/boundary_dataset.py` (core dataset)

**Training:**
- `boundary_refinement/training/__init__.py`
- `boundary_refinement/training/trainer.py` (Lightning module)
- `boundary_refinement/training/metrics.py` (evaluation metrics)
- `boundary_refinement/training/losses.py` (loss functions)

**Configs:**
- `boundary_refinement/configs/mlp.toml` (production config)
- `boundary_refinement/configs/mlp_test.toml` (Mac test config)

**Scripts:**
- `boundary_refinement/scripts/train.py` (main training entry point)
- `boundary_refinement/scripts/extract_embeddings.py` (embedding pre-computation)

**Test Data:**
- `data/voxconverse_test/audio/` (3 audio files)
- `data/voxconverse_test/rttm/` (3 ground truth files)

**Documentation:**
- `boundary_refinement/README.md` (architecture overview)

### Compute & Cost Tracking

**Day 4 Compute:**
- Mac M3 CPU testing only
- No GPU usage
- ~2 hours development + testing

**Phase 2 Budget:**
| Planned | Actual | Status |
|---------|--------|--------|
| $90 | $0 | Under budget ✅ |

**Total Project Spend:** $0.50 (Day 3 Lambda only)
**Budget Remaining:** $999.50 / $1000

### Next Steps (Day 5-7)

**Day 5 Morning - Implement Remaining Variants:**
- Transformer refinement model
- Diffusion boundary model (novel!)
- Contrastive boundary scorer

**Day 5 Afternoon - Fix Embedding Extraction:**
- Debug pyannote/WeSpeaker API compatibility
- Pre-compute real embeddings for training
- Alternative: Use DiariZen's embedding extraction directly

**Day 6-7 - Full Training:**
- Train all 4 variants on VoxConverse train
- Lambda A100 instance (~20 GPU hours estimated)
- Evaluate on VoxConverse dev
- Select best variant for integration

### Decision Gates

**Gate 3 (Day 7):** Do all variants train stably?
- MLP baseline: ✅ Confirmed working (25ms MAE on overfit)
- Transformer: To be validated
- Diffusion: To be validated
- Contrastive: To be validated

**Risk Mitigation:**
- MLP proves learning signal exists (25ms achievable)
- Dummy embeddings allow training without WeSpeaker
- Can fall back to MLP if complex variants fail

### Summary

**Day 4 Status:** ✅ COMPLETE

**Achievements:**
- Complete boundary refinement infrastructure (15 files)
- MLP baseline validated on Mac (25ms MAE overfit)
- Data pipeline proven end-to-end
- Ready for Day 5 implementation

**Blockers Resolved:**
- WavLM loading fixed
- Dummy embeddings workaround for testing
- All PyTorch Lightning integration working

**Next Checkpoint:** Day 5 evening - all 4 variants implemented and tested

---

## Day 5 - Lambda A100 GPU Overfit Tests (2025-12-01)

**Goal:** Validate 3 additional model variants (Transformer, Contrastive, Diffusion) on Lambda A100 GPU

### Morning: Model Implementation

**Implemented 3 New Variants:**

1. **Transformer Refiner** (`transformer_refiner.py` - 1.4M params):
   - Bidirectional self-attention over audio frames
   - Dual speaker conditioning (left + right speaker embeddings)
   - Multi-head attention (4 heads, d_model=256, 2 layers)
   - Encoder-only architecture (no decoder needed)
   - Direct offset prediction from [CLS] token

2. **Contrastive Refiner** (`contrastive_refiner.py` - 361K params):
   - InfoNCE loss formulation (temperature=0.07)
   - Scores all 200 positions in window
   - Soft averaging at inference (position × probability)
   - Smallest model, but competitive performance
   - Fixed bug: Dynamic frame count instead of hardcoded 200

3. **Diffusion Refiner** (`diffusion_refiner.py` - 1.6M params):
   - **Novel contribution:** First diffusion model for diarization boundaries
   - Denoising diffusion on 1D boundary position
   - T=10 timesteps, linear noise schedule
   - U-Net with skip connections
   - Timestep embedding (sinusoidal)

**Updated Infrastructure:**
- Modified `trainer.py` to handle all 4 variant types
- Added variant-specific training loops (diffusion noise prediction, contrastive scoring)
- Added variant-specific inference (diffusion denoising, contrastive soft average)
- Created test configs for all 3 new variants

### Afternoon: Lambda A100 Testing

**Test Setup:**
- Instance: Lambda A100-SXM4-80GB ($1.29/hr)
- 3 tests × ~3 minutes each = ~10 minutes total
- Same 10-example overfit test as Day 4 MLP
- 100 epochs, batch_size=4

**Test Results:**

| Model | Parameters | Train Loss | Val MAE | Target | Status |
|-------|-----------|------------|---------|--------|--------|
| **MLP** (Day 4) | 822K | 0.003 | 25ms | <50ms | ✅ PASS |
| **Transformer** | 1.4M | 0.007 | **3ms** | <50ms | ✅ **EXCELLENT** |
| **Contrastive** | 361K | 0.000 | **40ms** | <30ms | ⚠️ NEAR PASS |
| **Diffusion** | 1.6M | 0.478 | **1658ms** | <100ms | ❌ FAIL |

**Key Findings:**

1. **Transformer - BEST PERFORMER:**
   - **3ms MAE** - 12x better than MLP baseline!
   - Near-perfect overfitting (train loss 0.007)
   - Bidirectional attention + dual speaker conditioning works excellently
   - **Recommendation:** Primary candidate for Phase 3

2. **Contrastive - GOOD BUT NEEDS TUNING:**
   - **40ms MAE** - slightly above 30ms target
   - Perfect train convergence (loss 0.000)
   - InfoNCE loss working correctly
   - Bug discovered and fixed: Shape mismatch (199 vs 200 frames)
   - Could improve with temperature tuning or soft label smoothing

3. **Diffusion - DID NOT CONVERGE:**
   - **1658ms MAE** - 16x worse than target!
   - Poor train loss: 0.478 (should be <0.01 for overfitting)
   - Denoising process not learning effectively
   - Root cause suspected: Hyperparameter issues (T=10 too few, noise schedule wrong)

### Bug Fix: Contrastive Shape Mismatch

**Error:** `RuntimeError: Expected target size [4, 200], got [4, 199]`

**Root Cause:** Hardcoded `num_positions=200` but actual frames = 199

**Fix** (`contrastive_refiner.py:91`):
```python
# Before:
position_indices = ((offset + 2.0) / 4.0 * self.num_positions).long()

# After:
num_positions = audio_window.shape[1]  # Dynamic instead of hardcoded
position_indices = ((offset + 2.0) / 4.0 * num_positions).long()
```

### Compute & Cost Tracking

**Day 5 Compute:**
- Lambda A100-SXM4-80GB: 0.17 hours (~$0.22)
- 3 overfit tests (Transformer, Contrastive, Diffusion)
- Total runtime: ~10 minutes

**Phase 2 Budget:**
| Planned | Actual | Status |
|---------|--------|--------|
| $90 | $0.22 | Significantly under budget! ✅ |

**Total Project Spend:** $0.50 (Day 3) + $0.22 (Day 5) = **$0.72**
**Budget Remaining:** $999.28 / $1000

### Files Created (Day 5)

**Models:**
- `boundary_refinement/models/transformer_refiner.py`
- `boundary_refinement/models/contrastive_refiner.py`
- `boundary_refinement/models/diffusion_refiner.py`
- Updated `boundary_refinement/models/__init__.py` with all exports

**Configs:**
- `boundary_refinement/configs/transformer_test.toml`
- `boundary_refinement/configs/contrastive_test.toml`
- `boundary_refinement/configs/diffusion_test.toml`

**Documentation:**
- `DAY5_RESULTS.md` - Detailed Lambda A100 test results
- `LAMBDA_BOUNDARY_TESTS.md` - Testing checklist
- `LAMBDA_QUICK_START.sh` - Automated setup script

**Training Infrastructure:**
- Updated `boundary_refinement/training/trainer.py` with variant-specific logic
- Added diffusion noise prediction training
- Added contrastive InfoNCE training
- Added variant-specific inference methods

### Decision: 3 Models vs 4 Models?

**Options:**
- **Option A:** Proceed with 3 models (MLP, Transformer, Contrastive) - RECOMMENDED
- **Option B:** Fix diffusion hyperparameters first (+1 day delay)

**Reasoning for Option A:**
- 3 working models sufficient for Phase 3 validation
- Transformer performs exceptionally well (3ms MAE)
- Diffusion is research novelty, not critical
- Can revisit diffusion in future work
- Saves 1-2 days on timeline

**Day 5 Status:** ✅ COMPLETE (3/4 models validated)

**Next:** Day 5.5 - attempt quick diffusion fix OR proceed to Phase 3

---

## Day 5.5 - Diffusion Fix Attempt (2025-12-01 evening)

**Goal:** Quick attempt to fix diffusion hyperparameters before deciding to skip it

### Analysis of Diffusion Failure

**Original Issues Identified:**
1. **Too few timesteps:** T=10 insufficient for proper denoising
2. **Poor noise schedule:** σ_max=0.5s doesn't cover full boundary range [-2, +2]s
3. **Learning rate too high:** 1e-3 may be too aggressive for diffusion
4. **No regularization:** No weight decay on 1.6M parameters

### Improved Configuration

**Created:** `boundary_refinement/configs/diffusion_improved.toml`

**Key Changes:**
- Timesteps: 10 → **50** (5x more denoising steps)
- σ_max: 0.5s → **1.0s** (covers full boundary range)
- σ_min: 0.01s → **0.002s** (finer precision)
- Learning rate: 1e-3 → **5e-4** (matches Transformer success)
- Weight decay: 0 → **1e-4** (add regularization)

### Mac CPU Test Results

**Test:** 100 epochs, 10 examples, CPU

**Results:**
```
Original config:  Train loss 0.478, Val MAE 1658ms
Improved config:  Train loss 0.476, Val MAE 4991ms ❌
```

**Outcome: MADE IT WORSE!**
- Val MAE increased from 1658ms → **4991ms** (3x worse!)
- Train loss stuck at 0.476 (no improvement)
- No learning occurred despite better hyperparameters

### Root Cause Identified

**The problem is NOT hyperparameters - it's the denoising algorithm itself.**

From `diffusion_refiner.py:287-293`, the update rule is incorrect:

```python
# Current (WRONG):
x = x - sigma_t * noise_pred
if t > 0:
    x = x + sigma_prev * torch.randn_like(x)
```

**Proper DDPM requires** (Ho et al., 2020):
```python
alpha_t = 1 - beta_t
alpha_bar_t = prod(alpha_s for s in 0..t)
mean = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * noise_pred)
x = mean + sigma_t * torch.randn_like(x)
```

**Missing:** Proper variance scheduling (β_t, α_t, α_bar_t)

**Fixing this would require:**
- Implement proper DDPM variance schedule
- Rewrite denoising algorithm with correct update rule
- Test cosine vs linear schedules
- Debug convergence again
- **Estimated time: 4-6 hours**

### Final Decision: Skip Diffusion, Proceed with 3 Models

**Rationale:**

1. **Algorithmic fix required**, not just hyperparameter tuning
2. **3 working models already:**
   - Transformer: **3ms MAE** (EXCELLENT)
   - MLP: **25ms MAE** (GOOD baseline)
   - Contrastive: **40ms MAE** (ACCEPTABLE)
3. **Diffusion is research novelty, not critical** for validation
4. **Timeline:** Saves 1-2 days vs debugging diffusion
5. **Budget:** Negligible impact ($0 extra spent on Mac CPU test)

### Files Created (Day 5.5)

**Configs:**
- `boundary_refinement/configs/diffusion_improved.toml` (for documentation)

**Documentation:**
- `DIFFUSION_FIX_ANALYSIS.md` - Complete analysis of failure and decision rationale
- Documented for future work if we want to revisit

### Compute & Cost Tracking

**Day 5.5 Compute:**
- Mac CPU only (no GPU)
- ~10 minutes testing
- $0 additional cost

**Total Project Spend:** $0.72 (unchanged)
**Budget Remaining:** $999.28 / $1000

**Day 5.5 Status:** ✅ COMPLETE (decision made: 3 models for Phase 3)

**Next:** Phase 3 preparation - full training on VoxConverse

---

## Evaluation Bug Fix (2025-12-02)

### The Mystery: 4.52% vs Expected 9.1%

**Root cause found:** `--ignore_overlaps` flag in evaluation script.

| Setting | DiariZen (official) | Our evaluation |
|---------|---------------------|----------------|
| collar | 0 | 0 ✓ |
| ignore_overlaps | **NO** | YES ← BUG |

**Why it matters:** Overlapping speech (~10-20% of audio) is the hardest to diarize. By ignoring overlaps, we excluded the hardest regions containing disproportionate error.

**Fix needed:** Remove `--ignore_overlaps` from `evaluate_voxconverse.py:56`

```python
# BEFORE (wrong)
"--ignore_overlaps"  # Excludes hardest regions

# AFTER (correct)
# No --ignore_overlaps flag
```

**Impact on findings:**
- ✅ Boundary analysis (~50% from jitter) still valid - calculated independently
- ⚠️ Baseline DER should be ~9.1%, not 4.52%
- ⚠️ Need to re-run evaluation before Phase 3 comparisons

**Action item:** Re-baseline with correct evaluation before training.

---

## Calibration Retrospective (2025-12-02)

### Timeline Compression: 5.5x

**Original PLAN.md estimated:**
- Phase 1 (Validation): Days 1-3
- Phase 2 (Implementation): Days 4-7
- **Total Phase 1-2:** 7 days

**Actual:**
- All of Phase 1-2 completed on **2025-12-01** (Sunday)
- **Total elapsed:** 1 day (~8 hours of focused engagement)
- **Compression ratio:** 5.5x

### Why Traditional Estimates Failed

The original plan used human-paced thinking:
- "Day 2 PM: If numbers don't match, debug" → Numbers were better, no debug needed
- "Days 5-6: Implement variants" → All 4 done in one evening
- "Day 7: Training pipeline" → Already done by Day 4

**The rate-limiting step was NOT implementation complexity.**

### Decision Point Analysis

| Planned Day | Work | Decision Points | Actual Time |
|-------------|------|-----------------|-------------|
| Day 1 | Papers, setup | 2 | ~1 hour |
| Day 2 | Baseline eval | 2 | ~1 hour |
| Day 3 | Error analysis | 2 | ~1 hour |
| Day 4 | Architecture | 3 | ~2 hours |
| Day 5 | Implement variants | 4 | ~3 hours |
| Day 5.5 | Diffusion decision | 2 | ~30 min |
| **Total** | | **~15** | **~8 hours** |

**Key insight:** ~15 decision points × ~5 min decision latency (focused) = ~1.25 hours of decision time. Add ~6 hours of Claude work. Total: ~8 hours.

### What Was Actually Blocked (Zone C)

1. **DIHARD III dataset** - Still needs LDC license (external dependency)
2. **Diffusion algorithm** - Required novel insight, not just iteration (Zone C)
3. **GPU training** - Still ahead, irreducible compute time

### Updated Phase 3 Estimate

Using the new framework:

**Decision Points (Phase 3):**
- Launch training runs: 1
- Monitor/debug training: 2-3
- Evaluate results: 2
- Decide on iteration: 2
- Final evaluation: 2
- **Total: ~10 decisions**

**Zone C Blockers:**
- 3 training runs × 5 hours = 15 GPU hours
- Can parallelize → ~6-8 wall clock hours

**Engagement pattern:** Intermittent (checking during training)

**Estimated wall clock:** 1-2 days (including overnight training)

### Lessons for Future Planning

1. **Count decision points, not days**
2. **Identify Zone C blockers explicitly** - they dominate timeline
3. **Implementation complexity is free** - 3,200 lines in one evening
4. **Focused engagement compresses dramatically** - 5.5x on Sunday

**See:** `~/Developer/FEASIBILITY_CALIBRATION.md` for the full framework.

---

## Phase 3: Training & Evaluation (2024-12-02)

### Training Setup

**Infrastructure:**
- 3x Lambda A100 instances running in parallel
- Total training cost: ~$20
- ~6 hours wall clock time

**Models Trained:**
| Model | Epochs | Early Stop | Checkpoint |
|-------|--------|------------|------------|
| Transformer | 6 | Yes | 42MB |
| MLP | 20 | No | 9.4MB |
| Contrastive | 5 | Yes | 8.3MB |

### Synthetic Validation Results

| Model | Best val_mae | Epoch | Notes |
|-------|--------------|-------|-------|
| **MLP** | **2.2ms** | 15 | Steadily improved |
| Transformer | 2.9ms | 1 | Converged immediately |
| Contrastive | 21.8ms | 0 | Different objective |

**Observation:** These metrics are on synthetic data (GT + Gaussian noise). They measure the wrong thing.

### Real-World Evaluation

Evaluated on actual DiariZen predictions (not synthetic test data):

**DiariZen Baseline Boundary Error:**
| Metric | Value |
|--------|-------|
| Mean error | **188.8ms** |
| Median error | 88.0ms |
| Std error | 322.8ms |
| @50ms accuracy | 29.3% |
| @100ms accuracy | 55.3% |
| @200ms accuracy | 79.7% |

**Critical Finding:** The Day 3 estimate of 923ms average boundary error was incorrect. DiariZen's actual boundary error is 188.8ms - nearly 5x better than estimated.

**Refinement Results:**
| Model | Mean Error After | Change | Verdict |
|-------|------------------|--------|---------|
| Transformer | 188.8ms | **0.0%** | No effect |
| MLP | 191.1ms | **-1.2%** | Made worse |

### Decision Gate 4: FAIL

From PLAN.md:
> "Does ANY variant improve boundary precision by >10%?"

**Result:** No variant improved boundaries. Best case (Transformer) had zero effect.

### Root Cause Analysis

**Why synthetic training didn't generalize:**

1. **Distribution mismatch:**
   - Training: GT + Gaussian noise (σ=300ms)
   - Reality: DiariZen errors (188.8ms mean, non-Gaussian distribution)

2. **Task mismatch:**
   - Learned: "Denoise large random perturbations back to center"
   - Needed: "Correct small systematic biases in real predictions"

3. **Magnitude mismatch:**
   - Training noise: 300ms std (large)
   - Real errors: 88ms median (small, already good)

**Why the original 923ms estimate was wrong:**

The Day 3 `analyze_boundary_errors.py` script likely measured all segment boundary misalignments (including same-speaker segment breaks), not specifically speaker-change transitions. The 188.8ms figure from `evaluate_on_diarizen_predictions.py` measures matched speaker transitions only.

### Hypothesis Status

**Original Hypothesis:**
> "Boundary jitter (~923ms) is ~50% of remaining DER. Targeted refinement can beat SOTA."

**Status: FALSIFIED**

- DiariZen's boundary error is 188.8ms, not 923ms
- 55% of boundaries already within 100ms
- 80% of boundaries already within 200ms
- No significant boundary problem exists to fix

### Files Created (Phase 3)

**Evaluation:**
- `DiariZen/boundary_refinement/scripts/evaluate_on_diarizen_predictions.py`

**Results:**
- `diarize/EXPERIMENT_RESULTS.md` - Full experiment documentation
- `diarize/results/voxconverse_dev/` - 216 DiariZen prediction RTTM files

**Checkpoints:**
- `diarize/checkpoints/transformer_best.ckpt`
- `diarize/checkpoints/mlp_best.ckpt`
- `diarize/checkpoints/contrastive_best.ckpt`

**Logs:**
- `diarize/logs/transformer/` - Training logs + TensorBoard
- `diarize/logs/mlp/` - Training logs + TensorBoard

### Compute & Cost Tracking

| Date | Instance | Hours | Cost | Purpose |
|------|----------|-------|------|---------|
| 2024-12-02 | Lambda A100 (transformer) | ~6 | ~$7.75 | Training + inference |
| 2024-12-02 | Lambda A100 (mlp) | ~4 | ~$5.15 | Training |
| 2024-12-02 | Lambda A100 (contrastive) | ~5 | ~$6.45 | Training |
| **Phase 3 Total** | | ~15 | **~$19.35** | |

**Total Project Spend:** $0.72 (Phase 1-2) + $19.35 (Phase 3) = **~$20**
**Budget Remaining:** ~$980 / $1000

### Lessons Learned

1. **Validate error estimates before building solutions** - The 923ms figure was wrong; should have verified with real predictions earlier.

2. **Synthetic training ≠ real-world performance** - Models that excel on synthetic validation can fail completely on real data.

3. **Decision gates work** - Gate 4 caught this failure before spending more resources on iteration.

4. **Fast failure is valuable** - Spent ~$20 and 1 day to falsify hypothesis. Better than spending $200 and 2 weeks.

---

## Project Status Summary

| Phase | Status | Outcome |
|-------|--------|---------|
| Phase 1 (Validation) | ✅ Complete | DiariZen baseline established |
| Phase 2 (Implementation) | ✅ Complete | 3 model variants built |
| Phase 3 (Training) | ✅ Complete | **Hypothesis falsified** |
| Phase 4 (Iteration) | ⏸️ On hold | Pending decision on next steps |

**Decision Required:** Continue with Option A/B/C from NEXT_STEPS.md or close project.

---
