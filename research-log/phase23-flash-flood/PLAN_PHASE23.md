# Phase 23: Flash Flood (Diversity & Sampling)

## Objective
Determine the "Upper Bound" of solvability for the hard HumanEval problems by scaling test-time compute (Sampling).
Compare **Unstructured Diversity** (Baseline Sampling) vs. **Structured Diversity** (Fractal/Sketch Sampling).

## Core Question
"Does varying the **Plan** (Sketch) yield more diverse/correct solutions than just varying the **Code** tokens?"

## Experiment Design

### Dataset
Representative Set (15 hard problems).

### Method A: Baseline Sampling (Unstructured)
- **Prompt:** Standard Signature + Docstring.
- **Generation:** $N=50$ samples per problem.
- **Temperature:** 0.8 (High diversity).
- **Metric:** Pass@1, Pass@10, Pass@50 (using unbiased estimator).

### Method B: Fractal Sampling (Structured)
- **Step 1:** Generate $N=50$ **Sketches** (Plans). ($T=0.8$)
- **Step 2:** For each Sketch, generate **1 Code implementation** (Guided). ($T=0.2$ - rely on plan for diversity).
- **Metric:** Pass@1, Pass@10, Pass@50.

## Implementation
- `flash_flood_sampling.py`: Script to run both modes.
- Use `vLLM` batching for speed.
- `humaneval_harness.py`: execution logic (reused).

## Success Criteria
- **Existence Proof:** Do we solve the "Unsolvable 5" (HumanEval/10, 32, 33, 39, 129)?
- **Comparative Advantage:** Does Fractal Sampling achieve higher Pass@k than Baseline? (i.e., is the latent space of *Plans* better to search than the latent space of *Code*?)
