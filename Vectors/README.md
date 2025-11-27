# Fractal Engine Vectors

High-level directions for the Fractal Falsifiable Diffusion Engine.

---

## Vector 1: The "Flash Flood" Decoder (Speed SOTA)

Autoregressive models generate tokens sequentially. The Fractal Engine separates a tiny autoregressive Manager (roots) from a massively parallel renderer (chunks/characters), enabling parallel expansion of many roots at once. Goal: achieve ~10,000 tokens/second on consumer hardware via hierarchical parallelism.

**Status / Experiments:** Phases 4–6 implemented a working Manager+Fractal renderer on Shakespeare with shared weights across levels and parallel root expansion, but we have not yet run focused speed benchmarks or large‑scale throughput tests aimed at the 10k tokens/sec target.

---

## Vector 2: The "Holographic" Learner (Data Efficiency SOTA)

Traditional LLMs relearn structure at every scale. Fractal models share weights across levels (class → function → line), learning structural isomorphisms once and reusing them. Goal: match or beat GPT-3.5–class performance with <1B parameters and ~10B tokens by exploiting hierarchical structure and curriculum training.

**Status / Experiments:** Phases 1–4 showed that a single shared‑weight transformer can act at multiple abstraction levels (roots, chunks, characters) and generalize across them on synthetic trees and Shakespeare; we have not yet pushed this to large‑scale, GPT‑3.5‑class data‑efficiency comparisons.

---

## Vector 3: The "Ouroboros" Reasoner (Reliability SOTA)

Chain-of-thought helps, but models still hallucinate inside their own traces. With an energy head, the Fractal Engine scores logical consistency at each step and can backtrack when energy is high. Goal: near zero-hallucination behavior on verifiable domains (math, code, contracts) by rejecting high-energy reasoning transitions.

**Status / Experiments:** Phases 7–10 trained a GSM8K verifier (Ouroboros) on chain‑of‑thought traces and used it for selection; it failed to beat the baseline (≈50% vs 52% Pass@1) and often assigned low energy to confident hallucinations, leading to the pivot toward hard verification via execution.

---

## Vector 4: Alignment and Governance Engine

Use the energy head as a general-purpose verifier for normative constraints, not just factual correctness. Train energy on (input, output, compliant/violation) pairs for safety, style, security, licensing, and policy adherence. Goal: make the fractal verifier a built-in governance layer that can sit in front of any generator.

**Status / Experiments:** No dedicated governance experiments yet; Phase 10’s failure on GSM8K highlighted that text‑only energy heads are fragile and likely need to be combined with explicit rules or process reward models, which informs how a future governance engine should be built.

---

## Vector 5: Fractal World Models and RL "Dreamer"

Extend the text-only hierarchy to multi-scale world models: high-level plans (roots), mid-level transitions (chunks), low-level actions or states (fine). Energy becomes a learned physics / rules-violation detector for trajectories. Goal: use fractal diffusion + energy to power planning, imagination rollouts, and safe RL.

**Status / Experiments:** Phase 5 implemented a “Dreamer‑style” vertical generation demo for text (roots → chunks → characters) with energy‑based rejection, but we have not yet extended the architecture to full world models or RL environments.

---

## Vector 6: Program Synthesis and Execution Loop

Treat the Fractal Engine as a neuro-symbolic compiler: roots are specs, mid-levels are typed skeletons, leaves are concrete code. Couple generation with actual execution and tests, and train energy on pass/fail outcomes. Goal: a code model that not only writes programs but internally learns to predict which candidates will pass tests.

**Status / Experiments:** Phases 11–13 built a differentiable Neural ALU (adder + shift‑and‑add multiplier with digital restoration) that extrapolates in bit‑length. Phases 14–15 rebooted Vector 6 on MBPP/HumanEval: a 1.5B code model with an execution‑trained critic and short GRPO fine‑tuning achieved a reproducible ~+6–7% Pass@1 gain over baseline using Best‑of‑k + ranking, confirming the execution‑loop thesis at small scale.

---

## Vector 7: Hierarchical Editing and Lifelong Learning

Move beyond from-scratch generation to surgical edits of large trees (codebases, documents) while preserving global structure. Use energy to enforce consistency when modifying subtrees and to identify unstable regions for targeted retraining. Goal: continual learning via local updates instead of full retrains, with structural guarantees.

**Status / Experiments:** No direct hierarchical‑editing experiments yet; all current work generates from scratch. However, the Neural CPU and code‑verification pipeline suggest a path toward edit‑time verification of local changes in codebases.

---

## Vector 8: Multi-Modal Fractals

Apply the same root → chunk → fine hierarchy across modalities: scenes → objects → pixels for vision, sections → motifs → samples for audio, or mixed text–image–code trees. Train energy on cross-modal coherence (e.g., caption ↔ image ↔ code). Goal: a single fractal backbone that can reason and verify across multiple modalities.

**Status / Experiments:** No multi‑modal experiments have been run yet; all current fractal and verification work is text‑ and code‑only.

---

## Vector 9: Data Curation and Scientific Instrumentation

Use the fractal energy signals to study datasets and training dynamics, not just to filter generations. Detect mislabeled or structurally inconsistent samples, quantify "structural richness" of domains, and run controlled falsification experiments on inductive biases and scaling. Goal: turn the Fractal Engine into a scientific instrument for understanding learning, not just a product model.

**Status / Experiments:** Throughout Phases 2–15 we have used energy scores and execution outcomes to diagnose failure modes (e.g., GSM8K verifier collapse, analog drift in soft accumulators, critic overfitting on MBPP), but we have not yet run a focused “data curation” study where energy is used systematically to clean or reweight training data.

---

# Scoring Rubric

## Dimensions (1-5 scale)

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Demo-ability** | 32% | How impressive/showable is a working demo? |
| **Falsifiability** | 20% | Can we definitively prove it works or fails? |
| **Cloud ROI** | 15% | If we spend $100-500, what do we get? |
| **Baseline Clarity** | 15% | Is there a clear "beat X on Y" or "first to do Z"? |
| **Reuses Phase 4-6** | 18% | Leverages existing Fractal Engine code? |

## Scoring Guide

### Demo-ability (32%)
- **5**: "Wow" factor - people immediately get it (e.g., generates working code from spec)
- **4**: Very impressive with minimal explanation
- **3**: Needs explanation but impressive when understood
- **2**: Technically interesting but hard to demo
- **1**: Only researchers would care

### Falsifiability (20%)
- **5**: Binary pass/fail (code runs, math correct, tests pass)
- **4**: Clear numeric metric (accuracy, pass@k)
- **3**: Measurable but somewhat subjective (perplexity, human eval)
- **2**: Qualitative comparison possible
- **1**: Vague or hard to measure

### Cloud ROI (15%)
- **5**: $100 gets decisive result
- **4**: $200-300 gets meaningful progress
- **3**: $500 gets meaningful progress
- **2**: $500 gets partial result, need more
- **1**: $500 barely scratches surface

### Baseline Clarity (15%)
- **5**: Clear existing benchmark (HumanEval, GSM8K, MBPP)
- **4**: Standard benchmark exists, comparison straightforward
- **3**: Can construct fair comparison with effort
- **2**: Comparison possible but not apples-to-apples
- **1**: No good baseline exists, defining new capability

### Reuses Phase 4-6 (18%)
- **5**: Direct extension of existing code (new data only)
- **4**: Minor architecture tweaks + new tokenizer
- **3**: New tokenizer/data but same model architecture
- **2**: Significant architecture changes needed
- **1**: Requires major architectural overhaul

---

# Vector Scores

## Combined Evaluation Summary

| Vector | Gemini | Claude | GPT-5.1 | **Avg** |
|--------|--------|--------|---------|---------|
| 1. Flash Flood | 4.82 | 3.84 | 4.53 | **4.40** |
| 2. Holographic | 3.04 | 3.53 | 3.35 | **3.31** |
| 3. Ouroboros | 4.04 | 3.52 | 4.06 | **3.87** |
| 4. Governance | 3.34 | 2.70 | 3.06 | **3.03** |
| 5. World Models | 3.36 | 3.34 | 3.34 | **3.35** |
| 6. Program Synth | 4.82 | 4.82 | 4.82 | **4.82** |
| 7. Hier. Editing | 3.47 | 3.03 | 2.85 | **3.12** |
| 8. Multi-Modal | 2.93 | 3.33 | 3.33 | **3.20** |
| 9. Data Curation | 2.52 | 3.24 | 3.39 | **3.05** |

### Detailed Scores by Dimension

| Vector | Evaluator | Demo | Falsify | Cloud | Baseline | Reuse | **Total** |
|--------|-----------|------|---------|-------|----------|-------|-----------|
| **1. Flash Flood** | Gemini | 5 | 5 | 5 | 5 | 4 | 4.82 |
| | Claude | 4 | 5 | 4 | 4 | 2 | 3.84 |
| | GPT-5.1 | 4 | 5 | 4 | 5 | 5 | 4.53 |
| **2. Holographic** | Gemini | 2 | 3 | 2 | 4 | 5 | 3.04 |
| | Claude | 3 | 4 | 2 | 5 | 4 | 3.53 |
| | GPT-5.1 | 3 | 4 | 3 | 4 | 3 | 3.35 |
| **3. Ouroboros** | Gemini | 2 | 5 | 5 | 5 | 5 | 4.04 |
| | Claude | 3 | 5 | 3 | 5 | 2 | 3.52 |
| | GPT-5.1 | 3 | 5 | 3 | 5 | 5 | 4.06 |
| **4. Governance** | Gemini | 2 | 3 | 5 | 3 | 5 | 3.34 |
| | Claude | 2 | 4 | 3 | 3 | 2 | 2.70 |
| | GPT-5.1 | 2 | 4 | 3 | 3 | 4 | 3.06 |
| **5. World Models** | Gemini | 5 | 4 | 2 | 2 | 2 | 3.36 |
| | Claude | 4 | 4 | 2 | 4 | 2 | 3.34 |
| | GPT-5.1 | 4 | 4 | 2 | 4 | 2 | 3.34 |
| **6. Program Synth** | Gemini | 5 | 5 | 5 | 5 | 4 | 4.82 |
| | Claude | 5 | 5 | 5 | 5 | 4 | 4.82 |
| | GPT-5.1 | 5 | 5 | 5 | 5 | 4 | 4.82 |
| **7. Hier. Editing** | Gemini | 4 | 3 | 5 | 2 | 3 | 3.47 |
| | Claude | 3 | 3 | 3 | 2 | 4 | 3.03 |
| | GPT-5.1 | 3 | 3 | 3 | 2 | 3 | 2.85 |
| **8. Multi-Modal** | Gemini | 5 | 2 | 1 | 4 | 1 | 2.93 |
| | Claude | 5 | 4 | 1 | 4 | 1 | 3.33 |
| | GPT-5.1 | 5 | 4 | 1 | 4 | 1 | 3.33 |
| **9. Data Curation** | Gemini | 1 | 2 | 5 | 1 | 5 | 2.52 |
| | Claude | 2 | 4 | 5 | 1 | 5 | 3.24 |
| | GPT-5.1 | 2 | 4 | 5 | 2 | 5 | 3.39 |

### Consensus Ranking (Updated Nov 27, 2025)

**Weights**: Demo 32%, Falsify 20%, Cloud 15%, Baseline 15%, Reuse 18%

1. **Vector 6: Program Synthesis** (Avg: 4.82) ⭐ — **VALIDATED.** Unanimous top pick. +6.6% Pass@1 proven.
2. **Vector 1: Flash Flood Decoder** (Avg: 4.40) — Strong demo potential, speed multiplier for Best-of-k
3. **Vector 3: Ouroboros Reasoner** (Avg: 3.87) ⚠️ — Text-only energy failed; pivot to hard verification needed
4. **Vector 5: World Models** (Avg: 3.35) — Compute-intensive RL requirements
5. **Vector 2: Holographic Learner** (Avg: 3.31) — Data efficiency is expensive to prove
6. **Vector 8: Multi-Modal** (Avg: 3.20) — High demo-ability, prohibitive compute
7. **Vector 7: Hierarchical Editing** (Avg: 3.12) — Lacks standard benchmarks
8. **Vector 9: Data Curation** (Avg: 3.05) — Cheap to run but hard to demo
9. **Vector 4: Governance Engine** (Avg: 3.03) ⚠️ — Blocked by Vector 3 failure; needs rethink

---

## Individual Evaluations

- **[Gemini CLI Agent Evaluation](./GEMINI_EVALUATION.md)**
- **[Claude (Opus 4) Evaluation](./CLAUDE_EVALUATION.md)**
- **[GPT-5.1 Evaluation](./GPT-5.1-EVAL.md)**

