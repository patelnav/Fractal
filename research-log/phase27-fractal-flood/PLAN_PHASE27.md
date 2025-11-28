# Phase 27: Fractal Flood (The High-Speed Engine)

## Objective
Combine **Vector 6 (Fractal Coder)** and **Vector 1 (Flash Flood)** to create the "Fastest Code Generator".
We use the **Sketch** not for reasoning, but as a **Structural Seed** to enable parallel decoding.

## The Architecture: "Sketch-Guided Flash Flood"
1.  **Stage 1: AR Sketch (Sequential, Fast)**
    -   Model generates a concise "Skeleton" (e.g., function signatures + comments).
    -   Length: ~50-100 tokens.
    -   Time: ~1-2s (AR).
2.  **Stage 2: Flash Flood (Parallel, Instant)**
    -   Model expands the Skeleton into Full Code.
    -   Length: ~1000 tokens.
    -   Time: ~0.2s (Parallel).
    -   Total Latency: ~1.5s (vs ~25s for full AR).

## Experiment
Since we don't have a trained "Sketch-to-Code" Diffusion model, we will simulate the pipeline using **Pre-Computed Oracles** to measure the **Upper Bound of Throughput**.
We want to prove the **System Latency Profile**.

### Simulation Setup (Benchmark)
1.  **Task:** Generate a long file (e.g., 2000 tokens).
2.  **Baseline (AR):** Generate 2000 tokens.
3.  **Fractal Flood:**
    -   **Sketch:** Generate 100 tokens (AR).
    -   **Flood:** Expand to 2000 tokens (10 Parallel Steps).
4.  **Measure:** End-to-End Latency.

## Deliverable
A `latency_benchmark.py` script that plots:
-   `Latency` vs `Output Length`.
-   Showing the "Fractal Crossover Point" where the fixed overhead of Sketching is overtaken by the speed of Parallel Expansion.

## Hypotheses
-   For short code (<100 tokens), AR is faster.
-   For long code (>500 tokens), Fractal Flood is dominant.
