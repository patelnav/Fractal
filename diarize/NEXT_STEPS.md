# Next Steps - Phase 2: Boundary Refinement Implementation

**Current Status:** Phase 1 COMPLETE ✅
**Next Phase:** Implementation (Days 4-7)
**Goal:** Build 4 boundary refinement variants

---

## What We Learned in Phase 1

### ✅ Validation Complete
- **DER: 4.52%** on VoxConverse dev (better than expected)
- **~50% of error** comes from boundary jitter
- **Average boundary error:** 0.923 seconds (median: 0.426s)
- **Clear improvement path:** 0.9s → 0.2s could cut DER in half

### ✅ Decision Gates Passed
- Gate 1: Baseline verified (4.52% < 9.1% target) ✅
- Gate 2: Boundary jitter >25% (~50%!) ✅

**Conclusion:** Boundary refinement is THE right focus

---

## Phase 2 Overview - 4 Days

### Day 4 (Architecture Design & Data Prep)
**Morning:**
- Review and finalize boundary refinement interface design
- Study DiffSED architecture (diffusion for audio boundaries)
- Study Flow-TSVAD approach (generative refinement)

**Afternoon:**
- Create training data pipeline
- Extract boundary windows from DIHARD/AMI/VoxConverse
- Generate synthetic boundaries (add noise to ground truth)
- Set up Lambda instance for training

### Day 5-6 (Implementation)
Build 4 variants in parallel:

**Variant 1: MLP Baseline**
- Concatenate WavLM features + speaker embeddings
- Simple MLP predicts boundary offset
- Purpose: Sanity check - if this works, others should too

**Variant 2: Transformer Refinement**
- Bidirectional transformer over boundary window
- Cross-attention to left/right speaker embeddings
- Purpose: Standard attention-based approach

**Variant 3: Diffusion Boundary** (Our novel contribution!)
- Treat boundary position as coordinate to denoise
- Condition on audio features + speaker embeddings
- Multiple denoising steps (DiffSED-style)
- Purpose: Our main hypothesis

**Variant 4: Contrastive Boundary**
- Binary classifier: "Is this the correct boundary?"
- Train with positive/negative pairs
- Inference: slide window, find maximum
- Purpose: Alternative framing

### Day 7 (Training Pipeline)
- Set up training loop (PyTorch Lightning)
- Implement evaluation metrics:
  - Boundary offset MAE/RMSE
  - Accuracy at ±50ms, ±100ms, ±200ms
- Configure logging (wandb)
- Sanity training runs

---

## Architecture Design - To Be Detailed on Day 4

### Boundary Refinement Interface

**Input:**
```python
audio_window: Tensor       # 4s around boundary, shape [B, T, D]
speaker_left: Tensor       # Left speaker embedding [B, E]
speaker_right: Tensor      # Right speaker embedding [B, E]
initial_boundary: float    # Initial boundary position (seconds)
```

**Output:**
```python
refined_boundary: float    # Refined boundary position
confidence: float          # Optional confidence score
```

### Key Design Questions (Day 4)

1. **Window size:** 2s? 4s? 8s around boundary?
2. **Features:** WavLM layer 25? Multiple layers? Raw audio?
3. **Speaker embeddings:** From DiariZen's clustering? WeSpeaker?
4. **Training objective:**
   - Regression (predict offset directly)?
   - Classification (discretize into bins)?
   - Diffusion (denoise noisy boundary)?
5. **Augmentation:**
   - How much noise to add to synthetic boundaries?
   - Random crops, speed perturbation?

---

## Data Pipeline - Day 4 Afternoon

### Datasets to Use

**DIHARD III train:**
- Multi-domain diarization
- Challenging conditions
- Need to download (check licensing)

**AMI Corpus:**
- Meeting recordings
- Available on HuggingFace
- Good speaker transitions

**VoxConverse train:**
- Matches test domain
- YouTube videos
- Natural speech

**Synthetic boundaries:**
- Take ground truth boundaries
- Add Gaussian noise (±0.5s, ±1.0s)
- Create training pairs

### Data Format

```python
{
  'audio_window': Tensor,      # 4s audio centered on boundary
  'speaker_left_embed': Tensor,
  'speaker_right_embed': Tensor,
  'true_offset': float,        # Offset from window center to true boundary
  'metadata': {
    'file_id': str,
    'boundary_idx': int,
    'difficulty': str           # 'easy', 'medium', 'hard'
  }
}
```

### Pipeline Steps

1. **Extract all speaker transitions** from RTTM files
2. **Load audio** around each transition (±2s window)
3. **Extract speaker embeddings** using DiariZen's WeSpeaker model
4. **Create synthetic examples** by adding noise
5. **Save to HDF5/WebDataset** for fast training

**Estimated dataset size:**
- DIHARD III: ~10k boundaries
- AMI: ~20k boundaries
- VoxConverse: ~5k boundaries
- Synthetic (2x each): ~70k total
- **Total: ~105k training examples**

---

## Compute Budget - Phase 2

| Task | GPU Hours | Cost |
|------|-----------|------|
| Data preprocessing | 5 | $6 |
| Training V1 (MLP) | 5 | $6 |
| Training V2 (Transformer) | 10 | $13 |
| Training V3 (Diffusion) | 20 | $26 |
| Training V4 (Contrastive) | 10 | $13 |
| Debugging/iteration | 20 | $26 |
| **Total Phase 2** | **~70** | **~$90** |

**Budget after Phase 2:** ~$910 remaining

---

## Success Criteria - Phase 2

By end of Day 7, we should have:

✅ 4 trained boundary refinement models
✅ All models converging (loss decreasing)
✅ MLP baseline shows learning signal (MAE < 0.5s)
✅ Training pipeline ready for Phase 3 evaluation

**Decision Gate 3 (Day 7):** Do variants train stably?
- YES → Proceed to Phase 3 (evaluation)
- NO → Debug, simplify architectures

---

## Phase 3 Preview - Evaluation

**Day 8-9:** Full training (if needed)
**Day 10:** Evaluate all variants on DIHARD dev
- Boundary metrics (MAE, accuracy@50ms, etc.)
- Integrated DER (plug into DiariZen pipeline)
- Select best variant

**Day 11:** Deep analysis
- Where does it help? Where does it fail?
- Error patterns, visualizations

---

## Open Questions for Day 4

1. **Data availability:**
   - Can we access DIHARD III train? (check licensing)
   - Is AMI train easily accessible?

2. **Computational feasibility:**
   - Can we extract 105k boundary windows in reasonable time?
   - Storage requirements for preprocessed data?

3. **Architecture details:**
   - Review DiffSED paper for diffusion architecture
   - Check if Flow-TSVAD has any released details

4. **Baseline comparison:**
   - Do we need a "no refinement" baseline for comparison?
   - Or is DiariZen's 0.923s error sufficient baseline?

---

## Files to Review Before Day 4

### Papers to Re-read (Morning)
- **DiffSED** (Bhosale et al., AAAI 2023)
  - Diffusion for sound event boundary detection
  - Architecture reference for Variant 3
- **Flow-TSVAD** (Chen et al., 2024)
  - Flow matching for boundary refinement
  - Only 2 steps needed for good results!

### Code References
- DiariZen's WavLM feature extraction
- DiariZen's speaker embedding extraction (WeSpeaker)
- DiffSED architecture (if code available)

---

## Preparation Checklist

Before starting Day 4:
- [ ] Read DiffSED paper (focus on architecture)
- [ ] Read Flow-TSVAD paper (focus on methodology)
- [ ] Check DIHARD III download/license
- [ ] Check AMI Corpus on HuggingFace
- [ ] Review DiariZen's feature extraction code
- [ ] Set up development environment (can develop locally first)

---

## Current Status Files

**Documentation:**
- `LOG.md` - Days 1-3 complete
- `PHASE1_COMPLETE.md` - Phase 1 summary
- `LAMBDA_FIXES.md` - All setup fixes documented
- `NEXT_STEPS.md` - This file

**Results:**
- `results/voxconverse_dev/*.rttm` - 216 predictions
- `results/voxconverse_dev/detailed_boundary_analysis.txt` - Boundary analysis

**Data:**
- VoxConverse dev: Downloaded, 216 files
- VoxConverse annotations: Downloaded
- DIHARD III: Not yet downloaded
- AMI Corpus: Not yet downloaded

---

## When Ready to Start Day 4

**Ping me with:** "Start Day 4" or "Begin Phase 2"

**I will:**
1. Help design the boundary refinement interface
2. Set up the data pipeline
3. Download/prepare training datasets
4. Implement the 4 variant architectures in parallel

**Timeline:** 4 days (Day 4-7) to complete implementation
**Next checkpoint:** Day 7 evening - verify all variants train stably

---

## Key Reminder

**We have strong validation:** 50% of error is boundary-related, average error 0.923s

**Our goal:** Reduce boundary error by 4x (0.9s → 0.2s)
**Expected impact:** DER from 4.52% → 2-3%
**This would be:** Major SOTA improvement

**Let's build it!** 🚀
