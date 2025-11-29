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

---

## Competitor Benchmark

Benchmarked our neural repair against popular heuristic JSON repair tools.

### Competitors Tested

| Tool | Language | Monthly Downloads |
|------|----------|-------------------|
| json-repair | Python | 9.2M |
| fast-json-repair | Python/Rust | - |
| jsonrepair | JavaScript | 2.9M/week |

### Test Corpus

87 test cases across 7 error categories:

| Category | Count | Examples |
|----------|-------|----------|
| structural | 6 | Missing/extra braces and brackets |
| punctuation | 7 | Missing/extra commas and colons |
| quotes | 6 | Single quotes, unquoted keys, smart quotes |
| values | 6 | Python constants (`True`/`False`/`None`) |
| llm_specific | 7 | Markdown fences, prose, comments |
| multi_error | 5 | Multiple errors per document |
| synthetic | 50 | Random corruption via our engine |

### Results

```
Tool                     Parse%  Semantic%  Speed(ms)      Edits
------------------------------------------------------------
json-repair               98.9%      92.0%       0.02        4.2
fast-json-repair         100.0%      74.7%       0.00       13.2
jsonrepair-js             94.3%      86.2%      34.66        3.3
Neural (ours)             70.1%      66.7%      21.35        6.1
```

### By Category

```
Category         json-repair  fast-json  jsonrepair-js  Neural
---------------------------------------------------------------
llm_specific          100.0%     100.0%         85.7%    85.7%
multi_error           100.0%     100.0%        100.0%    40.0%
punctuation           100.0%     100.0%         71.4%    85.7%
quotes                100.0%     100.0%        100.0%    33.3%
structural             83.3%     100.0%         66.7%    66.7%
synthetic             100.0%     100.0%        100.0%    80.0%
values                100.0%     100.0%        100.0%    16.7%
```

### Analysis

**Why Neural Underperforms (70.1% vs 98.9%):**

The neural model was trained on a narrow distribution:
- ✅ Missing/extra commas, colons, brackets (punctuation class)
- ❌ Python constants (`True`→`true`, `None`→`null`)
- ❌ Quote normalization (single→double, smart→straight)
- ❌ LLM artifacts (markdown fences, prose wrapping)
- ❌ Multi-error documents

**Key Weaknesses:**
| Category | Neural % | Issue |
|----------|----------|-------|
| values | 16.7% | Never trained on constant normalization |
| quotes | 33.3% | Never trained on quote handling |
| multi_error | 40.0% | Trained on single-error only |

**Speed:** Neural is ~10,000x slower (21ms vs 0.02ms) due to model inference overhead.

### Conclusions

1. **On training distribution:** Neural achieves 97.6% (excellent)
2. **On production distribution:** Neural achieves 70.1% (poor)
3. **Heuristics win for syntax repair:** json-repair's rule coverage handles all edge cases
4. **Neural niche:** Semantic-level repair where context understanding matters

### Recommendations

1. **Expand training data:** Add Python constants, quote handling, LLM patterns
2. **Hybrid approach:** Heuristic pre-processing + neural for structural issues
3. **Target different problems:** Schema conformance, semantic repair, completion
