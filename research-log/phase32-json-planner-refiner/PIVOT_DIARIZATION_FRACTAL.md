# Diarization × Fractal Architecture: Deep Analysis

## Executive Summary

**Question:** Can Fractal's hierarchical neural architecture improve diarization quality enough to compete with ElevenLabs Scribe?

**Short answer:** Technically plausible, but high-risk research with uncertain payoff.

**Recommendation:** Validate with a focused 2-week spike before committing.

---

## Current State: diarize.io

### What You've Built
- **Production pipeline:** Parakeet-TDT v3 (ASR) + NeMo MSDD (diarization)
- **Optimizations:** Oracle VAD (45% speedup), Sortformer v2 (2.3% transition rate)
- **Infrastructure:** GCP Cloud Run + GPU, full product with payments

### What You've Tried (and Failed)

| Experiment | Result | Why It Failed |
|------------|--------|---------------|
| DiariZen | 0.93s boundary error | 60% worse than NeMo |
| Pyannote Community-1 | 2.62s boundary error | 4.5x worse than NeMo |
| Flatten-and-Group | +12.5% transitions | Can't fix upstream errors |
| DiarizationLM | Exploratory | LLM can't fix acoustic errors |
| GPU Phrase Boosting | 25% correction | Only helps known entities |

### The Core Problem

**NeMo MSDD achieves 0.58s average boundary error** - this is the BEST you found.

But 0.58s is still bad enough to cause:
- Mid-sentence speaker transitions
- Words assigned to wrong speaker
- User-visible quality issues

**Root cause:** Diarization boundaries are placed BEFORE actual speaker changes. No downstream algorithm can fix this.

---

## ElevenLabs Scribe: The Benchmark

### What Scribe Likely Does Better

| Capability | Your Pipeline | Scribe (Likely) |
|------------|---------------|-----------------|
| Architecture | Modular (VAD → ASR → Diarization → Assignment) | End-to-end joint model |
| Training data | Public datasets | Proprietary + massive scale |
| Boundary precision | 0.58s error | Unknown but "better" |
| Overlap handling | Clustering-based | Learned separation |

### Why Scribe Wins (Hypothesis)

1. **Joint optimization** - ASR and diarization trained together, not separately
2. **More data** - ElevenLabs has millions of hours of audio
3. **End-to-end** - No error propagation between stages
4. **Resources** - Well-funded team focused on this problem

---

## Fractal Architecture for Diarization

### The Core Insight

Fractal's key principle: **"Structure is the Signal"** - hierarchical processing where coarse levels guide fine levels.

**Current diarization (flat):**
```
Audio → VAD → Embeddings → Clustering → Word Assignment
        ↓         ↓            ↓              ↓
    (errors)  (errors)     (errors)       (errors compound)
```

**Fractal diarization (hierarchical):**
```
Level 0: Conversation    "Who's in this call?" (identify speakers)
            ↓
Level 1: Turn            "Who's speaking now?" (~5-10s windows)
            ↓
Level 2: Boundary        "Where exactly does speaker change?" (bidirectional refinement)
            ↓
Level 3: Word            "Which speaker said this word?" (final assignment)
```

### The Key Innovation: Bidirectional Boundary Refinement

**Current approach:** Single forward pass, boundary placed when embedding changes
**Fractal approach:** Given coarse boundary, refine using context on BOTH sides

```
Audio: [...speaker_A...]|[...speaker_B...]
                        ↑
                   Coarse boundary (±0.5s error)

Refinement:
- Look at 2s window around boundary
- Bidirectional attention (like diffusion "fill in the middle")
- Iteratively refine boundary position
- Use speaker embeddings as conditioning
```

### Why This Might Work

1. **Boundary errors are systematic** - Current models place boundaries early
2. **Bidirectional context helps** - Knowing what comes AFTER helps place boundary
3. **Iterative refinement** - Multiple passes can correct errors
4. **Conditioning on speakers** - Know WHO you're separating, not just THAT there's a change

### Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Speaker Inventory (Causal/AR)                      │
│ - Process full audio once                                    │
│ - Identify distinct speaker embeddings                       │
│ - Output: Set of speaker prototypes                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Coarse Segmentation (Causal)                       │
│ - Quick forward pass                                         │
│ - Place approximate boundaries (±0.5s)                       │
│ - Assign speakers to segments                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Boundary Refinement (Bidirectional/Diffusion)      │
│ - For each coarse boundary:                                  │
│   - Extract 2-4s window around boundary                      │
│   - Condition on adjacent speaker embeddings                 │
│   - Iteratively refine boundary position                     │
│ - Output: Precise boundaries (target: ±0.1s)                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Word Assignment (Deterministic)                    │
│ - Given precise boundaries + word timestamps                 │
│ - Simple intersection-based assignment                       │
│ - No ambiguity if boundaries are precise                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Honest Assessment

### Technical Feasibility: 6/10

**Pros:**
- Hierarchical approach is principled
- Bidirectional refinement is novel for diarization
- Could work on domain-specific data (architecture conversations)

**Cons:**
- Requires significant research (not just engineering)
- Need to train audio models (different from text)
- May not beat end-to-end approaches like Scribe

### Time & Resources

| Phase | Duration | Resources |
|-------|----------|-----------|
| Proof of concept | 2-4 weeks | 1 engineer + GPU |
| Working prototype | 2-3 months | 1-2 engineers + A100 |
| Production quality | 3-6 months | Team + infrastructure |

**Minimum viable experiment:** 2 weeks to test if bidirectional refinement improves boundary accuracy on your existing test set.

### Risk Factors

1. **Research risk** - May not work at all
2. **Data risk** - Need audio training data (harder than text)
3. **Compute risk** - Audio models are expensive to train
4. **Competition risk** - ElevenLabs keeps improving
5. **Opportunity cost** - 3-6 months not building product features

---

## Alternative Strategies

### Option A: Accept Commodity Diarization

**Strategy:** Use best available API (Scribe, AssemblyAI, Deepgram) and differentiate on:
- Speaker naming/identification (your domain expertise)
- Transcript enhancement (summaries, search, highlights)
- Vertical features (architecture-specific)
- Price/UX

**Pros:** Fast, low risk, focus on product
**Cons:** No technical moat, margin pressure

### Option B: Hybrid Improvement

**Strategy:** Keep your pipeline but add targeted fixes:
- Train a small boundary refinement model on YOUR error patterns
- Use LLM for post-hoc correction of obvious errors
- Build domain-specific fine-tuning (architecture vocabulary)

**Pros:** Incremental improvement, uses existing infrastructure
**Cons:** May not close gap with Scribe

### Option C: Full Fractal Research

**Strategy:** Build the hierarchical diarization system described above

**Pros:** Potential for real differentiation if it works
**Cons:** High risk, long timeline, may fail

### Option D: Acquisition Target

**Strategy:** Build enough traction to be acquired by a company that needs diarization

**Pros:** Exit without solving the hard problem
**Cons:** Limited upside, depends on market

---

## Recommended Path

### Phase 1: Spike (2 weeks)

**Goal:** Test if bidirectional refinement can improve boundary accuracy

**Approach:**
1. Take 10-20 audio samples with ground truth boundaries
2. Extract 2s windows around each boundary
3. Train a small model to predict precise boundary location
4. Measure: Can we improve from 0.58s → 0.2s error?

**Success criteria:** >50% reduction in boundary error on test set

**Failure mode:** Proceed to Option A (commodity + differentiation)

### Phase 2: Decision Point

If spike succeeds:
- Commit to 2-3 month prototype
- Need A100 access for training
- Parallel: Keep shipping product features

If spike fails:
- Accept Scribe is better
- Focus on value-add features
- Consider Scribe API integration for quality-critical use cases

---

## Questions to Answer Before Starting

1. **Do you have ground truth data?** Need audio + precise speaker boundaries for training
2. **Do you have GPU budget?** A100 time for audio model training
3. **What's the business case?** If diarization improves 50%, does retention/conversion improve?
4. **What's the opportunity cost?** What features won't you build during this research?
5. **Can you tolerate failure?** 50%+ chance this doesn't beat Scribe

---

## Appendix: Why Fractal Worked for Text (and Might for Audio)

### Text (JSON Repair)

| Factor | Finding |
|--------|---------|
| Structure | Hierarchical (document → objects → fields → characters) |
| Errors | Predictable (missing quotes, brackets, commas) |
| Solution | Diffusion-based denoising |
| Result | Works but heuristics are better (98.9% vs 42%) |

### Audio (Diarization)

| Factor | Finding |
|--------|---------|
| Structure | Hierarchical (conversation → turns → utterances → frames) |
| Errors | Systematic (boundaries placed early) |
| Solution | Bidirectional boundary refinement |
| Result | **Unknown - needs testing** |

**Key difference:** For audio, there's no simple heuristic that solves the problem. Current best (NeMo MSDD at 0.58s error) is genuinely improvable. This is where neural could add value.

---

## Conclusion

**The Fractal diarization approach is technically plausible but unproven.**

Unlike JSON repair (where heuristics won), diarization has no simple solution. The hierarchical, bidirectional refinement approach could work.

**But:** It's a research project with uncertain outcome. Don't commit without validating the core hypothesis in a 2-week spike.

**The honest question:** Are you willing to spend 3-6 months on research that might fail, when you could be shipping product features and using Scribe as a benchmark?
