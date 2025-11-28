# Phase 29: Fractal Initialization Proof

## Objective
Prove that **Sketch Initialization** enables sub-linear convergence in Parallel Decoding.

## The Experiment
Compare convergence speed of Jacobi Decoding under two conditions:
1.  **Naive Init:** Initialize with Padding (Baseline).
2.  **Fractal Init (Sketch):** Initialize with a "Rough Draft".
    -   **Synthetic Sketch:** Take Ground Truth and **Mask 50% of tokens** (e.g., every even token is replaced with PAD).
    -   This simulates a "Skeleton" where structure is known but details are missing.

## Hypothesis
-   **Naive:** Convergence = Linear ($N$ steps).
-   **Fractal:** Convergence = Sub-linear (e.g., $N/2$ or $log(N)$ steps).
    -   The model should fill in the gaps between valid tokens in parallel.
    -   Example: `def ____(n):` -> `def fib(n):` (1 step).

## Metrics
-   **Steps to Convergence:** How many forward passes to reach 100% match with AR Ground Truth?
-   **Speedup Factor:** $Steps_{Naive} / Steps_{Fractal}$.

## Implementation
`sketch_jacobi.py`:
-   Model: `Qwen/Qwen2.5-1.5B-Instruct`.
-   Task: Fibonacci Function (~50 tokens).
-   Run Naive.
-   Run Sketch (50% Masked).
-   Run Sketch (Random Noise - Control).
