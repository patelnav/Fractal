# Phase 26: Flash Flood at Scale

## Objective
Demonstrate **Extreme Parallel Generation** speedups using the Flash Flood (Vector 1) algorithm.
Show that we can generate massive structured outputs (e.g., 100x100 grids) in constant ($O(1)$) forward passes, whereas standard Transformers are linear ($O(N)$).

## The Benchmark: "Mega-Grid"
**Task:** Generate a large 2D Grid of integers (CSV format).
- Size: $H 	imes W$ (e.g., 64x64 = 4096 tokens).
- Constraint: Values must be consistent (e.g., Row $i$, Col $j$ = $i+j$).
- This constraint is simple enough to be learned, but the *structure* requires generating 4096 tokens.

## Models
1.  **Baseline (Sequential):** Standard AR generation.
    - Steps = 4096.
2.  **Flash Flood (Parallel):**
    - **Step 1 (Root):** Generate dimensions/prompt.
    - **Step 2 (Flood):** Initialize full 64x64 canvas with noise/mask.
    - **Step 3 (Refine):** Run Denoising/Refinement passes.
    - Target Steps = 5-10.

## Implementation Details
Since we don't have a pre-trained 2D diffusion transformer, we will simulate the "Perfect Predictor" scenario to measure **System Throughput**.
- We will use a mock model (or a small real one) that supports:
    - `forward(input_ids)` -> `logits`
    - **KV-Cache** for AR.
    - **Parallel Masked Prediction** for Flash Flood.

**Wait, we need a real model to measure real inference time on GPU.**
We can use a standard `GPT-2` or `Qwen` (small) and force it to generate via:
1.  **AR Loop:** `for i in range(4096): model(x)`
2.  **Jacobi/Parallel Loop:** `for k in range(10): model(full_x)`
    - (Even if the output is garbage, the *timing* is real).

## Experiment
1.  **Hardware:** A100 (Lambda).
2.  **Script:** `benchmark_speed.py`.
    - Load Model (e.g., GPT2-Large or Qwen-1.5B).
    - **Run AR:** Measure time to generate 1024 tokens.
    - **Run Flash Flood:** Measure time to run 10 parallel passes on 1024 tokens.
    - **Calculate Speedup:** `Time(AR) / Time(FF)`.

## Expected Result
- AR: ~50 tokens/sec (Sequential). Total ~20s for 1000 tokens.
- FF: ~10 forward passes. Each pass takes ~0.05s (parallel). Total ~0.5s.
- **Speedup: ~40x - 100x.**

## Deliverable
A plot: `Tokens Generated` vs `Time`.
- AR: Linear slope.
- FF: Step function (finishes instantly).
