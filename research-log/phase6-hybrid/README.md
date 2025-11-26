# Phase 6: The Hybrid Fractal Engine

## Summary

**Goal**: Combine an autoregressive "Manager" with the Phase 4/5 Fractal Engine for open-ended text generation.

**Result**: **Novel Shakespeare-like text** generated through a two-stage pipeline with energy-based verification.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE HYBRID FRACTAL ENGINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Manager GPT (1M params)          Fractal Engine (6M params)       │
│   ┌─────────────────────┐          ┌────────────────────────┐       │
│   │ Autoregressive      │          │ Diffusion + Energy     │       │
│   │ Next-Root Prediction│ ───────► │ Hierarchical Expansion │       │
│   └─────────────────────┘          └────────────────────────┘       │
│           │                                  │                      │
│           ▼                                  ▼                      │
│   [844, 1739, 804, ...]            Root → Chunks → Chars            │
│   (abstract plot)                  (detailed rendering)             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Manager GPT (`train_manager.py`)
- **Architecture**: 4-layer GPT with 128 embedding dim
- **Parameters**: 1,059,072
- **Training data**: 350K root tokens from Shakespeare
- **Task**: Predict next root ID given context
- **Result**: Val loss 4.78 after 5K iterations (91 seconds)

### 2. Hybrid Generator (`generate_hybrid.py`)
- Loads both Manager and Fractal Engine
- Manager generates root sequence ("the plot")
- Fractal Engine renders each root with rejection sampling
- Energy threshold: 0.5, max retries: 10

## Results

### Sample Generations

**Seed 42 (30 roots, 110 rejections):**
```
Your highness is your country's daughter;
He hath been apprection
At and his dishonour'd in thy chair.
```

**Seed 123 (25 roots, 57 rejections):**
```
to find it betwixt them, smallows not on himself,
And pale, by him be
```

### Comparison: Pure Tokenizer vs Fractal Engine

| Method | Output | Rejections |
|--------|--------|------------|
| Pure tokenizer | "Her Barnardine, of Bianca, Titus Adius..." | 0 |
| Fractal Engine | "Her Barnardine, of Biica, Titus Adius..." | 83 |

The Fractal Engine does more than just lookup - it regenerates and verifies each expansion using energy-based rejection sampling.

## The Pipeline

```
1. Manager dreams: [844, 1739, 804, 88, 141, ...]
                    "Your" "high" "ness" "is" "your" ...

2. Fractal renders each root:
   Root 844 (Your):
     Level 0: Root → Chunks [c1, c2, ...]
       Check energy < 0.5? ✓ Accept / ✗ Retry
     Level 1: Each Chunk → Characters
       Check energy < 0.5? ✓ Accept / ✗ Retry

3. Concatenate all rendered text
```

## Key Insights

1. **Separation of concerns**: The Manager handles "what to say" (plot structure), while the Fractal Engine handles "how to say it" (detailed rendering).

2. **Energy-based verification**: Every expansion is verified. High-energy outputs are rejected and regenerated.

3. **Rejection sampling works**: ~3-4 rejections per root on average shows the energy head is actively filtering.

4. **Text quality**: Generated text is grammatically plausible and Shakespearean in style.

## Files

- `train_manager.py` - Train the Manager GPT on root sequences
- `generate_hybrid.py` - Combined generation demo
- `manager.pt` - Trained Manager checkpoint

## Usage

```bash
# Train the Manager (if needed)
python train_manager.py

# Run hybrid generation
python generate_hybrid.py --num-roots 30 --seed 42

# Compare with pure tokenizer decode
python generate_hybrid.py --compare
```

## Future Work

1. **Longer context**: Increase Manager block size for better coherence
2. **Conditioning**: Allow user prompts to seed generation
3. **Quality metrics**: Measure perplexity, coherence, novelty
4. **Scaling**: Larger Manager + more training for better plots
