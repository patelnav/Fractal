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

### Day 2 - 2025-12-03 (Tue)
**Goal:** Baseline evaluation

| Time | Planned | Actual | Notes |
|------|---------|--------|-------|
| AM | Run DiariZen on VoxConverse test | | |
| AM | Run DiariZen on DIHARD III dev | | |
| PM | Compare to published (9.1% VoxC, 14.5% DIHARD) | | |
| PM | Debug if numbers don't match | | |

**Deliverable:** Verified baseline DER numbers
**Status:** [ ] Not started

**DECISION GATE 1:** Match DiariZen baseline within 1%?
- [ ] PASS
- [ ] FAIL → Action: ___

---

### Day 3 - 2025-12-04 (Wed)
**Goal:** Error analysis - validate hypothesis

| Time | Planned | Actual | Notes |
|------|---------|--------|-------|
| AM | Break down DER: Miss/FA/Confusion | | |
| AM | Extract boundary errors specifically | | |
| PM | Calculate % of error from boundaries | | |
| PM | Visualize, listen to failure cases | | |

**Deliverable:** Error analysis with boundary quantification
**Status:** [ ] Not started

**DECISION GATE 2:** Boundary jitter >25% of error?
- [ ] YES (>25%) → Proceed
- [ ] MAYBE (15-25%) → Proceed with caution
- [ ] NO (<15%) → Pivot or abort

**Key Finding:** ___ % of error is boundary-related

---

## Phase 2: Implementation (Days 4-7)

### Day 4 - 2025-12-05 (Thu)
**Goal:** Architecture design & data prep

| Time | Planned | Actual | Notes |
|------|---------|--------|-------|
| AM | Design boundary refinement interface | | |
| AM | Review DiffSED architecture | | |
| PM | Create training data pipeline | | |
| PM | Setup Lambda instance | | |

**Deliverable:** Data pipeline ready, architectures designed
**Status:** [ ] Not started

---

### Day 5 - 2025-12-06 (Fri)
**Goal:** Implement variants (part 1)

| Time | Planned | Actual | Notes |
|------|---------|--------|-------|
| AM | V1: MLP baseline | | |
| AM | V2: Transformer refinement | | |
| PM | Continue implementation | | |
| PM | Verify forward pass works | | |

**Status:** [ ] Not started

---

### Day 6 - 2025-12-07 (Sat)
**Goal:** Implement variants (part 2)

| Time | Planned | Actual | Notes |
|------|---------|--------|-------|
| AM | V3: Diffusion boundary (DiffSED-style) | | |
| AM | V4: Contrastive boundary | | |
| PM | Debug, verify all variants | | |

**Deliverable:** 4 trainable models
**Status:** [ ] Not started

---

### Day 7 - 2025-12-08 (Sun)
**Goal:** Training pipeline

| Time | Planned | Actual | Notes |
|------|---------|--------|-------|
| AM | Training loop setup | | |
| AM | Evaluation metrics | | |
| PM | Sanity training runs | | |
| PM | Estimate full training time | | |

**Deliverable:** Training pipeline ready
**Status:** [ ] Not started

**DECISION GATE 3:** Variants train stably?
- [ ] PASS - MLP shows learning signal
- [ ] FAIL → Debug architectures

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
_(Add as we discover them)_

### Unexpected Findings
_(Things that surprised us)_

### Technical Debt / Future Work
_(Things to revisit later)_

---

## Compute Tracking

| Date | Instance | Hours | Cost | Purpose |
|------|----------|-------|------|---------|
| | | | | |

**Total spent:** $___
**Budget remaining:** $1000 - $___

---

## Time Tracking

| Phase | Planned | Actual | Variance |
|-------|---------|--------|----------|
| Phase 1 (Validation) | 3 days | | |
| Phase 2 (Implementation) | 4 days | | |
| Phase 3 (Training) | 4 days | | |
| Phase 4 (Iteration) | 3 days | | |
| **Total** | 14 days | | |
