# Plan: Vector 1 - Flash Flood Decoder

## Goal
Achieve >10,000 tokens/second generation speed by implementing the "Flash Flood" hierarchical parallel decoding strategy.

## Baseline Analysis
The current `generate_hybrid.py` (Phase 6) implementation is sequential:
1. Manager generates Roots (Sequential AR).
2. Loop over Roots:
   - Expand Root -> Chunks (Sequential Rejection Sampling)
   - Loop over Chunks:
     - Expand Chunk -> Chars (Sequential Rejection Sampling)

This effectively serializes the entire process, negating the parallel advantage of the fractal architecture.

## The "Flash Flood" Strategy
1. **Manager Phase:** Generate $R$ roots autoregressively (System 1).
2. **Level 0 Flood:** Expand all $R$ roots into $R \times C$ chunks in a single GPU batch.
   - Apply Best-of-K rejection sampling in parallel (generate $K$ candidates per root, pick lowest energy).
3. **Level 1 Flood:** Expand all $R \times C$ chunks into $R \times C \times L$ characters in a single GPU batch.
   - Apply Best-of-K rejection sampling in parallel.

## Step-by-Step Implementation

### 1. Baseline Benchmark
- Create `benchmark_sequential.py`.
- Measure Wall-Clock time for generating ~1000 tokens using the existing `render_root` loop.
- Calculate Tokens/Sec.

### 2. Flash Flood Decoder Class
- Create `flash_flood.py`.
- Implement `FlashFloodDecoder` class.
- **Method `expand_level_parallel`**:
  - Input: Batch of condition IDs (e.g., [R] roots).
  - Operation:
    - Replicate inputs $K$ times (for Best-of-K).
    - Run `FractalDiffusionModel` forward pass on $R \times K$ batch.
    - Compute Energy for all $R \times K$ outputs.
    - Select best candidate for each of the $R$ inputs.
  - Output: Batch of expanded tokens (e.g., [R, ExpansionSize] chunks).

### 3. Full Pipeline Integration
- Connect Manager -> Level 0 Flood -> Level 1 Flood.
- Handle padding/masking (some roots might produce padding chunks).

### 4. Optimization & Benchmarking
- Measure Tokens/Sec of the Flash Flood pipeline.
- Tune Batch Size and Best-of-K parameter ($K$).
- Compare against Baseline.

## Hypothesis
If $R=20$ (roots), $C=4$ (chunks/root), $L=16$ (chars/chunk) = 1280 chars.
- **Sequential:** $20$ L0 steps + $80$ L1 steps = 100 inference steps.
- **Flash Flood:** 1 L0 step + 1 L1 step = 2 inference steps.
- Speedup should be approx $50x$.

## Files to Create
- `research-log/phase16-flash-flood/benchmark_sequential.py`
- `research-log/phase16-flash-flood/flash_flood.py`
- `research-log/phase16-flash-flood/test_flash_flood.py`
- `research-log/phase16-flash-flood/RESULTS.md`
