# Phase 2: Shakespeare Fractal Diffusion + Hallucination Detection

## Hypothesis

1. **Training**: Discrete diffusion can learn natural language hierarchies (Shakespeare text)
2. **Verification**: Generation energy (Chen's Lemma 7) can detect hallucinations - the "Popperian Engine"

## Experimental Setup

### Hierarchical Tokenization
Built from tinyshakespeare (1.1M characters):

| Level | Description | Vocab Size | Coverage |
|-------|-------------|------------|----------|
| Level 2 (Fine) | Characters | 65 | 100% |
| Level 1 (Chunks) | 4-char blocks | 2049 (top 2048 + UNK) | 53.5% |
| Level 0 (Roots) | 4-chunk blocks | 2049 (top 2048 + UNK) | 22.0% |

**Total vocab**: 4163 tokens

### Model Architecture
- **Layers**: 6 transformer blocks
- **Embedding**: 256 dimensions
- **Heads**: 8 attention heads
- **Parameters**: 6.31M
- **Position Encoding**: Rotary Positional Embeddings (RoPE)
- **Attention**: Bidirectional (no causal mask)

### Training Configuration
- **Iterations**: 10,000
- **Batch Size**: 128
- **Learning Rate**: 3e-4 with 500-step warmup
- **Diffusion**: 100 timesteps, Poisson bit-flip noise

## Results

### Training Performance

| Iteration | Train Loss | Eval Loss |
|-----------|------------|-----------|
| 0 | 8.26 | 8.24 |
| 1000 | 0.66 | 0.65 |
| 5000 | 0.007 | 0.008 |
| 8500 | 0.004 | **0.001** (best) |
| 10000 | 0.014 | 0.003 |

**Training Verdict**: SUCCESS - Model learned the hierarchical structure.

### Hallucination Detection Test

Tested 200 samples (100 Root->Chunk, 100 Chunk->Char):
- Compare energy of **correct** expansion vs **random wrong** expansion
- Hypothesis: Wrong expansions should have higher energy

| Metric | Value |
|--------|-------|
| Correct Energy (mean) | 90,269.89 +/- 17,210.69 |
| Wrong Energy (mean) | 90,614.66 +/- 17,917.06 |
| Energy Gap | +344.77 +/- 2,422.94 |
| Detection Rate | **52.5%** |

**Hallucination Verdict**: FAILURE - Detection rate is random chance.

## Analysis

The energy metric failed because:

1. **High Variance**: The standard deviation of the gap (2,422) is 7x larger than the mean gap (344)
2. **UNK Saturation**: 46.5% of chunks and 78% of roots are UNK tokens, making many "random" expansions actually plausible
3. **Energy Integration**: The score integral may need different normalization or a different functional form

## Artifacts

- `training_log.txt` - Complete training output with loss curves
- `hallucination_test.txt` - Full hallucination detection test output
- `tokenizer_stats.txt` - Hierarchical tokenizer statistics

## Code References

- `fractal_shakespeare.py` - Hierarchical tokenization
- `run_shakespeare.py` - Training script with RoPE and energy calculation
- `test_hallucination.py` - Hallucination detection test

## Next Steps

Potential improvements to test:
1. Use BPE tokenization instead of fixed 4-char chunks to reduce UNK rate
2. Try different energy functionals (e.g., KL divergence instead of score integral)
3. Train on larger dataset to get better distribution coverage
