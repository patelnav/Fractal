# Phase 3: Contrastive Energy Training for Hallucination Detection

## Summary

**Goal**: Achieve >95% hallucination detection on BPE-to-character decompression task.

**Result**: **100% detection rate** using trained energy head.

## Background

Phase 2.5 failed to detect hallucinations (~50% = random chance) because:
1. The model learned a "lookup table" - perfect decompression but weak conditioning
2. Implicit energy (Chen's Lemma 7: ∫||score||² dt) couldn't distinguish valid from invalid pairs
3. The model treated any plausible character sequence as valid, ignoring the BPE token constraint

## The Fix: Contrastive Energy Training

Instead of computing expensive Chen Score at inference, we train an **Energy Head** to predict validity directly using contrastive pairs as supervision:

- **Correct pairs** (BPE token → its characters): target energy = 0
- **Wrong pairs** (BPE token → different token's characters): target energy = 1

This is "amortized inference" - the network learns to predict what the expensive physics calculation would output.

## Architecture Changes

```python
# Added to DecompressionDiffusion model
self.energy_head = nn.Sequential(
    nn.Linear(n_embd, n_embd // 2),
    nn.GELU(),
    nn.Linear(n_embd // 2, 1)
)

# Forward now returns (logits, energy) when return_energy=True
def forward(self, x, t, attention_mask=None, return_energy=False):
    # ... transformer ...
    logits = self.head(h[:, 1:, :])
    if return_energy:
        energy = self.energy_head(h[:, 1:, :]).mean(dim=(1, 2))
        return logits, energy
    return logits
```

## Training Changes

Dual loss function:
```python
# Diffusion loss (denoising) - only on correct pairs
loss_diff = CrossEntropy(logits_correct, targets)

# Energy loss (contrastive) - on both correct and wrong pairs
loss_energy = MSE(energy_correct, 0) + MSE(energy_wrong, 1)

# Total
loss = loss_diff + lambda_energy * loss_energy
```

## Training Results

| Iteration | Eval Detection | Energy Loss |
|-----------|----------------|-------------|
| 0 | 48.2% | 1.01 |
| 500 | 78.2% | 0.39 |
| 1000 | 85.7% | 0.28 |
| 2000 | 88.2% | 0.25 |
| 3000 | 91.0% | 0.22 |
| 5500 | 92.0% | 0.20 |
| 8500 | 89.5% | 0.19 |

Peak training detection: **96%** (iter 8200)

## Final Test Results

```
============================================================
SUMMARY
============================================================

Test                                Detection Rate  Status
-----------------------------------------------------------------
Energy Head (Phase 3)               100.0%           SUCCESS (>95%)
Legacy: Energy (random wrong)       50.0%            FAILURE
Legacy: Cross-token (mismatched)    54.0%            FAILURE
```

### Energy Head Performance

| Metric | Value |
|--------|-------|
| Correct Energy | 0.0009 ± 0.04 (target: 0) |
| Wrong Energy | 0.9842 ± 0.09 (target: 1) |
| Correct < 0.5 threshold | 100.0% |
| Wrong > 0.5 threshold | 99.0% |
| Wrong > Correct | **100.0%** |

## Key Insights

1. **Implicit energy fails for discrete lookup tables**: When a model memorizes mappings perfectly, the diffusion score becomes nearly uniform - no "friction" to measure.

2. **Explicit contrastive training works**: By directly teaching the model "these pairs are wrong", we build the discriminator that implicit physics couldn't provide.

3. **Amortized inference is practical**: Instead of 50+ forward passes to integrate energy, we get the answer in a single forward pass with `return_energy=True`.

4. **The "Popperian Falsifier" is real**: We now have a mechanism to reject hallucinations - if `energy > 0.5`, the generation is invalid for that condition.

## Files

- `run_bpe_diffusion.py` - Training script with contrastive energy
- `test_hallucination.py` - Test suite with energy head detection
- `bpe_tokenizer.py` - BPE tokenizer (unchanged from Phase 2.5)
- `checkpoints/best_model.pt` - Trained model
- `hallucination_results.json` - Detailed test results

## Next Steps

Phase 3 validates the core mechanism. Potential extensions:
1. Scale to full Shakespeare text generation (not just BPE decompression)
2. Test on out-of-distribution prompts
3. Integrate rejection sampling: generate → check energy → reject if high
4. Explore multi-level hierarchy (tokens → sentences → paragraphs)
