# Phase 32: JSON Repair Engine - Results

## Summary

Built a neural JSON Repair Engine using the Universal Denoiser approach from Phase 31. After scaling to A100 and adding REINFORCE loss, achieved **97.6% parse success** - exceeding the 90% target.

**Key Finding:** Cross-entropy loss plateaus; REINFORCE with hard validators drives real improvement.

## Architecture

```
Broken JSON → Tokenize → Denoiser (8L Transformer) → Sample → Validate → Output
```

**Components:**
- `tokenizer_json.py`: JSON-specific tokenizer (vocab_size=105)
- `data_json.py`: Corruption engine with 7 corruption types
- `model_denoiser.py`: Bidirectional transformer denoiser
- `train_denoiser_a100.py`: A100-optimized training with REINFORCE
- `inference_repair.py`: Parser-in-the-loop repair algorithm

## Final Training Configuration (A100)

- Model: 8 layers, 8 heads, 512 embedding dim (~26M params)
- Training: 500K samples, batch_size=512, AMP fp16
- Loss: Cross-Entropy + REINFORCE (weight=0.1)
- REINFORCE reward: `json.loads()` success (binary 0/1)
- Hardware: Lambda Labs A100 40GB

## Benchmark Results

### Training Progression

| Phase | Parse@1 | Val Loss | Notes |
|-------|---------|----------|-------|
| Initial (CPU, 5K samples) | 36% | 0.8+ | Undertrained |
| CE-only (A100, epoch 30) | 83.2% | 0.03 | **Plateau** |
| CE + REINFORCE (epoch 31) | **97.6%** | 0.024 | Target exceeded |

### Comparison to Baselines

| Method | Parse@1 | Notes |
|--------|---------|-------|
| Do Nothing | 35.7% | Broken input |
| Heuristic Fixer | 87.9% | Rule-based |
| **Denoiser + REINFORCE** | **97.6%** | Neural, learned |

## The REINFORCE Insight

**Problem:** Cross-entropy optimizes token-level accuracy as a *proxy* for parse success:
- 99% token accuracy can still fail to parse (one wrong `}` breaks everything)
- CE loss dropped to 0.03 but parse success stuck at 83%
- The model was optimizing the wrong objective

**Solution:** REINFORCE with `json.loads()` as reward:
```python
reward = 1.0 if json.loads(sample) else 0.0
loss = ce_loss + lambda * -(reward - baseline) * log_prob(sampled_tokens)
```

**Result:** Parse success jumped from 83.2% to 97.6% in just ONE epoch.

## Key Learnings

1. **"Hard validators drive soft generalization"** (echoes Phase 14): When the true objective is non-differentiable (parser, compiler, tests), use it as a REINFORCE reward.

2. **Proxy metrics plateau, real metrics don't:** CE loss can reach near-zero while the real metric stays stuck. REINFORCE directly optimizes what you care about.

3. **REINFORCE is underutilized:** Many tasks have binary validators that are ignored during training. This principle applies to:
   - JSON/XML parsing
   - Code compilation
   - Mathematical correctness
   - Schema validation
   - Test suite passing

## Files

```
phase32-json-planner-refiner/
├── tokenizer_json.py       # JSON tokenizer
├── data_json.py            # Dataset & corruption engine
├── model_denoiser.py       # Denoiser + Energy head
├── train_denoiser.py       # Local training script
├── train_denoiser_a100.py  # A100 training with REINFORCE
├── inference_repair.py     # Repair loop
├── benchmark.py            # Evaluation harness
├── checkpoints_reinforce/
│   ├── best_denoiser.pt    # 97.6% parse success model
│   └── config.json         # Model config
└── RESULTS.md              # This file
```

## Validation

The core hypothesis from JSON_REPAIR.md:
> With realistic corruption training and a parser in the loop, the denoiser+energy system can achieve ParseSuccess@1 ≥ 90%

**Status:** VALIDATED. Final result: 97.6% (target was 90%).

**Conclusion:** Phase 32 demonstrates that:
1. Bidirectional denoisers excel at repair tasks (as predicted by Phase 31)
2. REINFORCE with hard validators breaks through proxy-metric plateaus
3. The "Planner + Renderer" architecture benefits from verifiable reward signals
