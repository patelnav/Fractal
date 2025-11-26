# Phase 2.5: BPE Decompression Test

## Motivation

Phase 2 (Shakespeare next-char prediction) hallucination detection **FAILED** with 52.5% detection rate (random chance). Root cause: 78% UNK tokens made the energy metric meaningless.

Phase 2.5 tests the same hypothesis with a cleaner setup:
- **0% UNK coverage** via BPE tokenization trained on the same corpus
- **Deterministic ground truth**: each BPE token has exactly one correct character sequence
- **Simpler task**: BPE token → character sequence (decompression)

## Dataset

- **Source**: TinyShakespeare (1.1M characters)
- **BPE vocab**: 512 tokens (65 base chars + 447 merges)
- **Training samples**: 503,147 (one per BPE token occurrence in text)
- **Max sequence length**: 16 characters
- **UNK rate**: 0% (by construction)

## Model

- **Architecture**: Transformer with RoPE, bidirectional attention
- **Parameters**: 5.4M
- **Task**: Given BPE token ID, predict character sequence via discrete diffusion
- **Diffusion**: Poisson bit-flip noise on Boolean hypercube (Chen 2025)

## Training Results

```
Iterations: 8,500
Final Loss: 0.0000 (perfect convergence)
Best Eval Loss: 0.0000 at iter 7,000
Training Time: ~4.5 minutes on MPS
```

The model learned the BPE→chars mapping perfectly.

## Hallucination Detection Results

| Test | Detection Rate | Status |
|------|---------------|--------|
| Energy (random wrong chars) | 60.0% | PARTIAL |
| Diversity (generation variance) | 32.0% | FAILURE |
| Cross-token (mismatched BPE→chars) | 51.0% | FAILURE |

### Test Descriptions

1. **Energy Test**: Compare energy of correct vs random character sequences for the same BPE token
   - Correct Energy: 1656.40 ± 967.61
   - Wrong Energy: 1659.82 ± 970.06
   - Gap: 3.42 ± 13.40 (signal-to-noise ratio ~0.26)

2. **Diversity Test**: Generate multiple decompressions from same token, measure variance
   - Real tokens: 0.002 diversity (very consistent)
   - Fake tokens: 0.027 diversity (still very consistent)
   - Model is too deterministic to distinguish

3. **Cross-token Test**: Compare energy of correct pairing (token A → chars A) vs wrong pairing (token A → chars B)
   - Correct: 2167.03 ± 915.64
   - Wrong: 2128.33 ± 845.42
   - Gap: -38.69 ± 1194.51 (wrong direction, high variance)

## Analysis

Despite achieving 0% UNK and perfect training loss, hallucination detection still fails. Key observations:

1. **Perfect memorization problem**: The model memorized all 512 BPE→chars mappings exactly. This makes the energy landscape nearly flat for in-distribution queries.

2. **Cross-token failure (51%)**: The model doesn't distinguish between correct and incorrect BPE→chars pairings. This suggests the conditioning mechanism isn't creating token-specific energy wells.

3. **Energy variance dominates signal**: Gap of 3.42 vs std of 13.40 means the energy difference is swamped by noise.

## Conclusions

The energy-based hallucination detection from Chen's paper may require:
- **Underfit models**: Perfect memorization eliminates the energy signal
- **Continuous data**: Discrete tokens may not preserve score function smoothness
- **Different energy formulation**: The integral of ||score||² may need modification for discrete diffusion

## Files

- `bpe_tokenizer.py`: Minimal BPE implementation with 0% UNK
- `run_bpe_diffusion.py`: Diffusion model + training
- `test_hallucination.py`: Three hallucination detection methods
- `checkpoints/best_model.pt`: Trained model
- `hallucination_results.json`: Detailed test results
