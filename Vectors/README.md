# Fractal Engine Vectors

High-level directions for the Fractal Falsifiable Diffusion Engine.

---

## Vector 1: The "Flash Flood" Decoder (Speed SOTA)

Autoregressive models generate tokens sequentially. The Fractal Engine separates a tiny autoregressive Manager (roots) from a massively parallel renderer (chunks/characters), enabling parallel expansion of many roots at once. Goal: achieve ~10,000 tokens/second on consumer hardware via hierarchical parallelism.

---

## Vector 2: The "Holographic" Learner (Data Efficiency SOTA)

Traditional LLMs relearn structure at every scale. Fractal models share weights across levels (class → function → line), learning structural isomorphisms once and reusing them. Goal: match or beat GPT-3.5–class performance with <1B parameters and ~10B tokens by exploiting hierarchical structure and curriculum training.

---

## Vector 3: The "Ouroboros" Reasoner (Reliability SOTA)

Chain-of-thought helps, but models still hallucinate inside their own traces. With an energy head, the Fractal Engine scores logical consistency at each step and can backtrack when energy is high. Goal: near zero-hallucination behavior on verifiable domains (math, code, contracts) by rejecting high-energy reasoning transitions.

---

## Vector 4: Alignment and Governance Engine

Use the energy head as a general-purpose verifier for normative constraints, not just factual correctness. Train energy on (input, output, compliant/violation) pairs for safety, style, security, licensing, and policy adherence. Goal: make the fractal verifier a built-in governance layer that can sit in front of any generator.

---

## Vector 5: Fractal World Models and RL "Dreamer"

Extend the text-only hierarchy to multi-scale world models: high-level plans (roots), mid-level transitions (chunks), low-level actions or states (fine). Energy becomes a learned physics / rules-violation detector for trajectories. Goal: use fractal diffusion + energy to power planning, imagination rollouts, and safe RL.

---

## Vector 6: Program Synthesis and Execution Loop

Treat the Fractal Engine as a neuro-symbolic compiler: roots are specs, mid-levels are typed skeletons, leaves are concrete code. Couple generation with actual execution and tests, and train energy on pass/fail outcomes. Goal: a code model that not only writes programs but internally learns to predict which candidates will pass tests.

---

## Vector 7: Hierarchical Editing and Lifelong Learning

Move beyond from-scratch generation to surgical edits of large trees (codebases, documents) while preserving global structure. Use energy to enforce consistency when modifying subtrees and to identify unstable regions for targeted retraining. Goal: continual learning via local updates instead of full retrains, with structural guarantees.

---

## Vector 8: Multi-Modal Fractals

Apply the same root → chunk → fine hierarchy across modalities: scenes → objects → pixels for vision, sections → motifs → samples for audio, or mixed text–image–code trees. Train energy on cross-modal coherence (e.g., caption ↔ image ↔ code). Goal: a single fractal backbone that can reason and verify across multiple modalities.

---

## Vector 9: Data Curation and Scientific Instrumentation

Use the fractal energy signals to study datasets and training dynamics, not just to filter generations. Detect mislabeled or structurally inconsistent samples, quantify "structural richness" of domains, and run controlled falsification experiments on inductive biases and scaling. Goal: turn the Fractal Engine into a scientific instrument for understanding learning, not just a product model.

---

# Scoring Rubric

## Dimensions (1-5 scale)

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Demo-ability** | 25% | How impressive/showable is a working demo? |
| **Falsifiability** | 20% | Can we definitively prove it works or fails? |
| **M2 Feasibility** | 15% | Can we prototype on M2 (64GB) before cloud? |
| **Cloud ROI** | 15% | If we spend $100-500, what do we get? |
| **Baseline Clarity** | 15% | Is there a clear "beat X on Y" or "first to do Z"? |
| **Reuses Phase 4-6** | 10% | Leverages existing Fractal Engine code? |

## Scoring Guide

### Demo-ability (25%)
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

### M2 Feasibility (15%)
- **5**: Full experiment fits in 64GB, trains in hours
- **4**: Full experiment possible, trains in 1-2 days
- **3**: Prototype possible, full run needs cloud
- **2**: Minimal prototype possible locally
- **1**: Requires cloud from the start

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

### Reuses Phase 4-6 (10%)
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
| 1. Flash Flood | 4.75 | 4.00 | 4.10 | **4.28** |
| 2. Holographic | 2.95 | 3.45 | 3.35 | **3.25** |
| 3. Ouroboros | 4.75 | 4.70 | 4.25 | **4.57** |
| 4. Governance | 3.80 | 3.20 | 3.10 | **3.37** |
| 5. World Models | 3.15 | 3.20 | 2.95 | **3.10** |
| 6. Program Synth | 4.60 | 4.50 | 4.50 | **4.53** |
| 7. Hier. Editing | 3.70 | 3.10 | 2.85 | **3.22** |
| 8. Multi-Modal | 2.65 | 3.05 | 3.45 | **3.05** |
| 9. Data Curation | 2.80 | 3.45 | 3.60 | **3.28** |

### Detailed Scores by Dimension

| Vector | Evaluator | Demo | Falsify | M2 | Cloud | Baseline | Reuse | **Total** |
|--------|-----------|------|---------|-----|-------|----------|-------|-----------|
| **1. Flash Flood** | Gemini | 5 | 5 | 4 | 5 | 5 | 4 | 4.75 |
| | Claude | 4 | 5 | 4 | 4 | 4 | 2 | 4.00 |
| | GPT-5.1 | 4 | 4 | 4 | 4 | 4 | 5 | 4.10 |
| **2. Holographic** | Gemini | 2 | 3 | 3 | 2 | 4 | 5 | 2.95 |
| | Claude | 3 | 4 | 3 | 2 | 5 | 4 | 3.45 |
| | GPT-5.1 | 3 | 4 | 3 | 3 | 4 | 3 | 3.35 |
| **3. Ouroboros** | Gemini | 4 | 5 | 5 | 5 | 5 | 5 | 4.75 |
| | Claude | 5 | 5 | 4 | 4 | 5 | 5 | 4.70 |
| | GPT-5.1 | 4 | 4 | 4 | 4 | 5 | 5 | 4.25 |
| **4. Governance** | Gemini | 3 | 3 | 5 | 5 | 3 | 5 | 3.80 |
| | Claude | 2 | 4 | 4 | 3 | 3 | 4 | 3.20 |
| | GPT-5.1 | 3 | 3 | 3 | 3 | 3 | 4 | 3.10 |
| **5. World Models** | Gemini | 5 | 4 | 2 | 2 | 2 | 2 | 3.15 |
| | Claude | 4 | 4 | 2 | 2 | 4 | 2 | 3.20 |
| | GPT-5.1 | 3 | 4 | 2 | 2 | 4 | 2 | 2.95 |
| **6. Program Synth** | Gemini | 5 | 5 | 4 | 4 | 5 | 4 | 4.60 |
| | Claude | 5 | 5 | 4 | 4 | 5 | 3 | 4.50 |
| | GPT-5.1 | 5 | 5 | 4 | 4 | 5 | 3 | 4.50 |
| **7. Hier. Editing** | Gemini | 4 | 3 | 5 | 5 | 2 | 3 | 3.70 |
| | Claude | 3 | 3 | 4 | 3 | 2 | 4 | 3.10 |
| | GPT-5.1 | 3 | 3 | 3 | 3 | 2 | 3 | 2.85 |
| **8. Multi-Modal** | Gemini | 5 | 2 | 1 | 1 | 4 | 1 | 2.65 |
| | Claude | 5 | 4 | 1 | 1 | 4 | 1 | 3.05 |
| | GPT-5.1 | 5 | 4 | 2 | 2 | 4 | 2 | 3.45 |
| **9. Data Curation** | Gemini | 1 | 2 | 5 | 5 | 1 | 5 | 2.80 |
| | Claude | 2 | 4 | 5 | 5 | 1 | 5 | 3.45 |
| | GPT-5.1 | 2 | 4 | 5 | 4 | 3 | 5 | 3.60 |

### Consensus Ranking

1. **Vector 3: Ouroboros Reasoner** (Avg: 4.57) — All evaluators agree this is a top pick
2. **Vector 6: Program Synthesis** (Avg: 4.53) — Universal agreement on high potential
3. **Vector 1: Flash Flood Decoder** (Avg: 4.28) — Strong demo potential, some disagreement on reuse
4. **Vector 4: Governance Engine** (Avg: 3.37) — Middle tier, valuable but niche
5. **Vector 9: Data Curation** (Avg: 3.28) — Cheap to run but hard to demo
6. **Vector 2: Holographic Learner** (Avg: 3.25) — Data efficiency is expensive to prove
7. **Vector 7: Hierarchical Editing** (Avg: 3.22) — Lacks standard benchmarks
8. **Vector 5: World Models** (Avg: 3.10) — Compute-intensive RL requirements
9. **Vector 8: Multi-Modal** (Avg: 3.05) — High demo-ability, prohibitive compute

---

## Individual Evaluations

- **[Gemini CLI Agent Evaluation](./GEMINI_EVALUATION.md)**
- **[Claude (Opus 4) Evaluation](./CLAUDE_EVALUATION.md)**
- **[GPT-5.1 Evaluation](./GPT-5.1-EVAL.md)**


