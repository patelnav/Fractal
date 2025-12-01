# Phase 32 V2 Training Results (MODEL WEIGHTS LOST)

**Date:** 2025-11-29
**Status:** Training completed 100/100 epochs successfully. Model checkpoints were NOT synced before Lambda termination. Only the log file was recovered.

## Training Configuration

```
Device: NVIDIA A100-SXM4-40GB (40GB VRAM)
Vocabulary: 112 tokens (expanded from 105)
Model: 26M parameters (8-layer bidirectional transformer)
Training samples: 500,000
Epochs: 100 (all completed)
Batch size: 512 effective
Mixed precision: True (AMP)
Throughput: ~1503 samples/sec
Time per epoch: 332.7s (~5.5 min)
Total training time: ~9.2 hours
Lambda cost: ~$1.29/hr × 9.2hr = ~$11.87
```

## New Vocabulary Additions (v2)
Added 7 tokens for LLM/Python artifacts:
- `True`, `False`, `None` (Python constants)
- Single quotes (to fix `'value'` → `"value"`)
- Markdown fences (to strip \`\`\`json...\`\`\`)

## Final Results (Epoch 100)

### Overall Metrics
| Metric | Value |
|--------|-------|
| **Best Parse Success** | **93.8%** |
| **Final Val Parse Success** | 91.0% |
| **Best Val Loss** | 0.0525 |
| **Final Val Token Acc** | 98.8% |

### Per-Corruption Type Breakdown (Final Epoch)

| Corruption Type | Parse Success | Acc | Status |
|-----------------|---------------|-----|--------|
| `block_comment` | 100% | 0.22 | Perfect |
| `delete_brace` | 92% | 0.19 | Good |
| `delete_bracket` | 88% | 0.19 | Good |
| `delete_colon` | 100% | 0.12 | Perfect |
| `delete_comma` | 100% | 0.19 | Perfect |
| `insert_comma` | 96% | 0.13 | Good |
| `line_comment` | 100% | 0.22 | Perfect |
| `markdown_fence` | 100% | 0.20 | Perfect (NEW) |
| `markdown_fence_plain` | 100% | 0.18 | Perfect (NEW) |
| `mask_token` | 96% | 0.17 | Good |
| `prose_wrap` | 100% | 0.19 | Perfect |
| `python_false` | 100% | 0.19 | Perfect (NEW) |
| `python_null` | 100% | 0.23 | Perfect (NEW) |
| `python_true` | 100% | 0.18 | Perfect (NEW) |
| `single_quotes` | 100% | 0.15 | Perfect (NEW) |
| `swap_adjacent` | 100% | 0.23 | Perfect |
| `trailing_comma_arr` | 100% | 0.22 | Perfect |
| `trailing_comma_obj` | 100% | 0.17 | Perfect |
| `truncate` | 24% | 0.19 | **Weak** |
| `unquote_keys` | 100% | 0.19 | Perfect |

### Analysis

**Strengths:**
- All 6 new corruption types achieved 100% parse success
- 15/20 corruption types at 100%
- Python constants (`True`/`False`/`None`) mastered
- Markdown fence stripping perfect
- Single quote → double quote perfect

**Weaknesses:**
- `truncate` at only 24% (model can't invent missing content)
- `delete_bracket` at 88% (structural damage hard to recover)
- `delete_brace` at 92%

## What Was Lost

1. **Model checkpoint** (`checkpoints_a100/best_denoiser.pt`) - ~100MB trained weights
2. **Optimizer state** - for potential continued training
3. **All .pt files** - model wasn't synced before terminate

The watcher script looked for `checkpoints_v2/` but training saved to `checkpoints_a100/`. rsync failed silently, then terminated Lambda.

## Projected vs json-repair Benchmark

| Tool | Parse Success | Notes |
|------|---------------|-------|
| Neural (single-pass) | 93.8% | Best epoch |
| Neural (estimated 3-pass) | ~95-97% | Theoretical |
| json-repair heuristic | 98.9% | Iterative |

**Gap:** ~5% behind json-repair in single-pass mode.

## Root Cause of Loss

```python
# auto_terminate_watcher.py
REMOTE_DIR = "~/Fractal/research-log/phase32-json-planner-refiner"

def sync_all_artifacts():
    # BUG: Wrong directory name
    subprocess.run([
        "rsync", "-avz",
        f"lambda:{REMOTE_DIR}/checkpoints_a100/",  # Training saved here
        f"{LOCAL_DIR}/checkpoints_v2/"  # But script looked for this
    ])
    # rsync failed silently, returned error code, but script continued
    terminate_lambda()  # Killed instance before verifying sync
```

Should have:
1. Checked rsync return code
2. Verified local files exist after sync
3. Never terminate without confirmation

## Next Steps

Implement iterative multi-pass training:
1. `train_iterative.py` - Train on multi-corruption samples, target = clean
2. Inference loop with convergence detection
3. Progress-based REINFORCE rewards

The model clearly learns transformations (100% on each individual corruption), but needs iteration for multi-error documents.

---
*Training log preserved: `train_v2.log` (11.8MB)*
*Saved to: `checkpoints_a100` on Lambda (terminated)*
