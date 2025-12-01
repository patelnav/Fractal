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

## Recommended Path (Recalibrated via Feasibility Zones)

### The Old (Wrong) Plan
- "2-week spike to validate"
- "Then decide whether to commit 3-6 months"
- This is **Zone A thinking disguised as prudence**

### What's Actually Zone A vs Zone C

| Blocker | Zone | Implication |
|---------|------|-------------|
| Implementation | A | "Build 5 variants" = 2-3 days total |
| Trying architectures | A | "Compare approaches" = parallel, not serial |
| GPU training | C | Irreducible, but can parallelize experiments |
| ElevenLabs data moat | C | Can't compete on general audio scale |
| Domain fine-tuning | A | YOUR architecture data might be enough |

### The Real Strategy: 5-Day Sprint, 5+ Experiments

| Day | Activity |
|-----|----------|
| **Day 1** | Build 4-5 boundary refinement variants in parallel: (1) MLP on embeddings, (2) Bidirectional transformer, (3) Diffusion-style iterative, (4) Contrastive loss, (5) Fine-tune existing diarizer |
| **Day 2** | Finish implementations, set up training pipeline |
| **Day 3-4** | Train all variants (Zone C - GPU time, but parallel) |
| **Day 5** | Evaluate all, pick winner, plan next iteration |

**This is what "iteration is cheap" actually means.** Don't do 1 careful experiment - do 5 fast ones.

### The Domain Moat Play

**Don't try to beat Scribe on general audio.** That's fighting their Zone C advantage (100M+ hours of training data).

**Instead:** Fine-tune on YOUR architecture domain audio + add boundary refinement.

If you have:
- 100+ hours of architecture conversations
- Ground truth speaker labels
- Domain-specific vocabulary

Then a smaller model specialized on YOUR domain can beat Scribe's general model *on your use case*.

### Success Criteria (After 5-Day Sprint)

- **Win:** Any variant achieves <0.3s boundary error (vs current 0.58s)
- **Partial:** Improvement to 0.4-0.5s, worth iterating
- **Fail:** No improvement → use Scribe, differentiate on features

### If Sprint Succeeds

Week 2: Iterate on winning approach, try 5 more variants
Week 3-4: Productionize, integrate with diarize.io pipeline
Week 5+: Ship and measure real-world quality

**Total timeline to production:** 4-6 weeks, not 3-6 months

### If Sprint Fails

Not a "failure" - it's information. Means boundary refinement isn't the lever.

Then:
- Use Scribe for quality-critical jobs
- Differentiate on features (speaker naming, summarization, search)
- Your pipeline for cost-sensitive jobs

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

## Conclusion (Feasibility-Calibrated)

**The Fractal diarization approach is Zone A: feels ambitious, actually routine.**

Unlike my initial assessment ("3-6 months, high risk"), the recalibrated view:

| My Initial Take | Recalibrated Reality |
|-----------------|---------------------|
| "2-week spike first" | 5-day sprint with 5 parallel experiments |
| "3-6 months to build" | 4-6 weeks to production |
| "High risk research" | Zone A exploration (cheap to try) |
| "Validate before committing" | Just try it - iteration is cheap |

**The only Zone C blockers:**
1. GPU training time (irreducible but parallelizable)
2. ElevenLabs' data moat (don't fight it - go domain-specific)

**Everything else is Zone A:** Implementation, architecture exploration, fine-tuning, evaluation.

**The honest question (recalibrated):** Why are we hesitating? The cost of trying is 5 days. The cost of NOT trying is wondering if we could have beaten Scribe on our domain.

**Just build 5 variants and see what works.**
