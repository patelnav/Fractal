# Phase 1: Synthetic Fractal Diffusion Test

## Hypothesis

Can a single set of neural weights learn discrete diffusion denoising at **two different abstraction levels** simultaneously, using only a conditioning token to distinguish between tasks?

This is the "Universal Refinement Hypothesis" - if true, it suggests scale-invariant learning is possible.

## Experimental Setup

### Synthetic Hierarchy
A 3-level toy hierarchy with deterministic 1-to-4 expansions:

```
Level 0 (Roots):  0-9   -> expands to 4 chunks
Level 1 (Chunks): 10-19 -> expands to 4 fine tokens
Level 2 (Fine):   20-29 -> terminal tokens
```

Example: Root `0` -> Chunks `[10, 11, 12, 13]` -> Fine tokens for each chunk

### Model Architecture
- **Layers**: 4 transformer blocks
- **Embedding**: 128 dimensions
- **Attention**: Bidirectional (no causal mask)
- **Time Embedding**: Sinusoidal + MLP projection
- **Vocab Size**: 30 tokens

### Training
- **Diffusion**: Discrete Poisson bit-flip noise (100 timesteps)
- **Task**: Given condition token + noisy targets, predict clean targets
- **Mixed Training**: Both Root->Chunk and Chunk->Fine tasks in same batches

## Results

| Metric | Value |
|--------|-------|
| Final Loss | ~0.02 |
| Recursive Accuracy | **100%** (200/200 tokens) |

The model correctly expanded all 10 roots through both levels to produce all 200 fine tokens.

## Verdict: SUCCESS

The hypothesis is **viable**. Shared weights can learn multi-level discrete diffusion.

## Artifacts

- `fractal_output.txt` - Complete training log
- `RESULTS.md` - Summary of results

## Code Reference

See `fractal_diffusion.py` in repository root.
