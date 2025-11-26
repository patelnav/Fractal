# Phase 7: Ouroboros Reasoner - Training Results

## Overview

**Objective**: Train an energy-based model to distinguish correct reasoning from incorrect reasoning, enabling self-verification of chain-of-thought outputs.

**Approach**: Contrastive energy learning where the model learns to output:
- Low energy (→0) for correct reasoning chains
- High energy (→1) for incorrect/hallucinated reasoning

## Architecture

| Component | Specification |
|-----------|---------------|
| Model | Bidirectional Transformer + Energy Head |
| Parameters | 63.6M |
| Layers | 12 |
| Embedding Dim | 512 |
| Attention Heads | 8 |
| Position Encoding | RoPE |
| Energy Head | MLP with Sigmoid (bounds output to [0,1]) |

## Training Data

| Dataset | Train Pairs | Val Pairs |
|---------|-------------|-----------|
| GSM8K (Math) | 7,473 | 1,319 |
| HumanEval (Code) | 124 | 31 |
| **Total** | **7,597** | **1,350** |

### Contrastive Pair Generation

Each pair consists of:
- **Context**: Question/prompt (shared)
- **Correct Target**: Ground truth solution
- **Wrong Target**: Perturbed incorrect solution

**Perturbation Types (Math)**:
1. `wrong_answer_big`: Change answer significantly (±100, ×10)
2. `remove_steps`: Remove half the reasoning steps
3. `contradictory`: Add self-contradicting statement
4. `scramble_numbers`: Replace numbers with wrong values

**Perturbation Types (Code)**:
1. `return_constant`: Replace return value with constant
2. `empty_body`: Replace function body with `pass`
3. `wrong_logic`: Add obviously wrong logic
4. `syntax_break`: Introduce syntax errors

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Hardware | NVIDIA A100-SXM4-40GB |
| Batch Size | 64 |
| Max Iterations | 5,000 |
| Learning Rate | 3e-4 (cosine decay to 3e-5) |
| Warmup | 200 iterations |
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Weight Decay | 0.1 |
| Precision | bfloat16 |
| Compile | torch.compile enabled |

## Results

### Final Metrics (Iteration 5000)

| Metric | Training | Validation |
|--------|----------|------------|
| Energy (Correct) | 0.014 | 0.234 |
| Energy (Wrong) | 0.991 | 0.715 |
| Detection Rate | **99.3%** | **80.4%** |
| Loss | 0.012 | 0.463 |

### Training Progression

| Iteration | Train Det | Val Det | Train Loss | Val Loss |
|-----------|-----------|---------|------------|----------|
| 0 | 52.7% | 51.7% | 0.500 | 0.500 |
| 500 | 78.1% | 74.7% | 0.368 | 0.411 |
| 1000 | 96.2% | 78.5% | 0.093 | 0.389 |
| 1500 | 98.5% | 80.6% | 0.166 | 0.405 |
| 2000 | 96.2% | 79.7% | 0.093 | 0.432 |
| 2500 | 98.1% | 80.1% | 0.055 | 0.450 |
| 3500 | 99.1% | 80.7% | 0.024 | 0.447 |
| 4500 | 99.3% | 80.4% | 0.012 | 0.463 |

### Key Observations

1. **Energy Separation Works**: The model successfully learns to assign low energy to correct solutions and high energy to incorrect ones.

2. **Train/Val Gap**: Significant overfitting visible (99% train vs 80% val detection). The model memorizes training perturbation patterns.

3. **Validation Plateau**: Val detection rate plateaus around 80% after ~1500 iterations despite continued training improvement.

4. **Energy Calibration**: On training data, energies are nearly perfect (0.01 vs 0.99). On validation, there's more overlap (0.23 vs 0.72).

## Comparison to Phase 3

Phase 3 (Contrastive Energy for hallucination detection) achieved 100% detection on its task. The key difference:

| Aspect | Phase 3 | Phase 7 |
|--------|---------|---------|
| Task | Factual statement verification | Reasoning chain verification |
| Context Length | Short statements | Long reasoning chains (256+ tokens) |
| Detection Rate | 100% | 80% |
| Data Size | ~1K examples | ~7.5K pairs |

Phase 7's lower validation performance likely stems from:
1. Longer sequences making discrimination harder
2. More subtle perturbations in reasoning vs factual errors
3. Validation set containing genuinely harder examples

## Critical Bug Fixed

**Problem**: Initial training showed model collapse with both energies converging to 0.5 (~50% detection, random chance).

**Root Cause**: Training used independent samples (different contexts for correct vs wrong). The model couldn't learn what makes a solution "wrong" without seeing it paired with the correct solution for the same problem.

**Solution**: Restructured to paired training (Phase 3 style):
```python
def compute_paired_loss(model, contexts, correct_targets, wrong_targets, ...):
    energy_correct, _ = model(contexts, correct_targets, ...)
    energy_wrong, _ = model(contexts, wrong_targets, ...)

    loss = F.mse_loss(energy_correct, torch.zeros_like(energy_correct)) + \
           F.mse_loss(energy_wrong, torch.ones_like(energy_wrong))

    detection_rate = (energy_wrong > energy_correct).float().mean()
    return loss, {'detection_rate': detection_rate, ...}
```

## Artifacts

| File | Description | Size |
|------|-------------|------|
| `checkpoints/ckpt.pt` | Trained model weights | 764 MB |
| `data/processed/train.npz` | Training data (paired format) | 11.2 MB |
| `data/processed/val.npz` | Validation data (paired format) | 2.0 MB |

## Future Improvements

To achieve >90% validation detection:

1. **Regularization**: Increase dropout, add weight decay
2. **Data Augmentation**: Generate more diverse perturbation types
3. **Harder Negatives**: Mine examples where model fails
4. **Ensemble**: Combine multiple models trained on different perturbations
5. **Curriculum Learning**: Start with obvious errors, gradually increase difficulty
6. **Contrastive Margin Loss**: Use ranking loss instead of MSE for better separation

## Conclusion

Phase 7 demonstrates that energy-based reasoning verification is feasible. The model achieves **80% accuracy** in distinguishing correct from incorrect reasoning chains on held-out validation data. While below the 90% target, this represents a significant improvement over random chance (50%) and validates the core approach.

The paired contrastive training paradigm (from Phase 3) is critical - without it, the model collapses to trivial solutions. Future work should focus on reducing overfitting and generating more challenging training examples.

---

*Training completed: 2025-11-26*
*Hardware: Lambda Labs A100 (40GB)*
*Duration: ~20 minutes*
*Cost: ~$0.40*
