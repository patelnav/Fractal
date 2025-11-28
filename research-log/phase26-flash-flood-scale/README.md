# Phase 26: Flash Flood Scale Benchmark

## Objective
Demonstrate the **extreme throughput potential** of the Flash Flood (Vector 1) parallel decoding algorithm compared to standard Autoregressive (AR) decoding on long sequences.

## Experiment
- **Hardware:** NVIDIA A100.
- **Model:** GPT-2 (Simulated inference cost for 1.5B parameters).
- **Method:**
    - **AR:** Standard sequential generation (`use_cache=True`). $N$ steps.
    - **Flash Flood:** Simulated parallel refinement (10 steps). $O(1)$ steps relative to length.
- **Lengths:** 128, 512, 1024, 2048 tokens.

## Results

| Length | AR Time (s) | AR TPS | FF Time (s) | FF TPS  | Speedup |
|--------|-------------|--------|-------------|---------|---------|
| 128    | 3.31        | 39     | 0.32        | 396     | **10x** |
| 512    | 12.97       | 39     | 0.31        | 1634    | **41x** |
| 1024   | 25.91       | 39     | 0.33        | 3128    | **79x** |
| 2048   | 51.76       | 39     | 0.55        | 3736    | **94x** |

## Analysis
1.  **Linear vs Constant:** AR generation time scales linearly with length (Sequence Length / ~40 TPS). Flash Flood generation time is roughly constant (~0.3-0.5s) because it always uses a fixed number of parallel refinement steps (10 steps), and the GPU handles the larger batch/sequence size in parallel.
2.  **Throughput Unlocked:** At 2048 tokens, we reach **~3700 Tokens/Sec**. This is orders of magnitude faster than human reading speed and enables "Instant UI" experiences for generating entire files, tables, or logs.
3.  **The Tradeoff:** This assumes the Flash Flood model *can* generate high-quality output in 10 steps. This benchmark measures the **Speed Ceiling**. The Quality floor remains to be proven (Phase 18/21/22 explored this).

## Conclusion
Flash Flood is **viable** as a high-speed engine for long-context generation. If we can constrain the search space (e.g., via Sketches from Phase 18), we can achieve both Quality and Speed.
