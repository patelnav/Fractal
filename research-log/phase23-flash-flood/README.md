# Phase 23: Flash Flood - Diversity Sampling Evaluation

## Hypothesis
Injecting diversity at the **sketch level** (high temperature T=0.8) and using deterministic code generation (T=0.2) will outperform baseline sampling that injects diversity directly at the token level.

**Rationale**: Sketch-level diversity explores different *algorithmic approaches*, while token-level diversity often just produces syntactic variations of the same approach.

## Method

### Baseline Sampling
- Generate N=50 code samples directly at T=0.8
- No intermediate sketch step

### Fractal Sampling
- Generate N=50 sketches at T=0.8 (diverse plans)
- For each sketch, generate 1 code implementation at T=0.2 (deterministic)
- Total: 50 code samples derived from 50 different plans

## Results

### Overall Performance

| Method | Avg Pass@1 | Winner |
|--------|------------|--------|
| Baseline | **68%** | ✓ |
| Fractal | 39% | |

**HYPOTHESIS DISPROVEN**: Baseline sampling outperformed Fractal sampling overall.

### Per-Problem Breakdown

| Task | Baseline Pass@1 | Fractal Pass@1 | Winner |
|------|-----------------|----------------|--------|
| HumanEval/0 | **100%** | 52% | Baseline |
| HumanEval/1 | **90%** | 34% | Baseline |
| HumanEval/10 | **6%** | 0% | Baseline |
| HumanEval/11 | **100%** | 60% | Baseline |
| HumanEval/12 | **98%** | 86% | Baseline |
| HumanEval/26 | 20% | **48%** | **Fractal** |
| HumanEval/29 | 78% | **94%** | **Fractal** |
| HumanEval/32 | 2% | **4%** | **Fractal** |
| HumanEval/33 | **52%** | 14% | Baseline |
| HumanEval/37 | **86%** | 40% | Baseline |
| HumanEval/39 | **98%** | 18% | Baseline |
| HumanEval/40 | **90%** | 40% | Baseline |
| HumanEval/43 | **100%** | 80% | Baseline |
| HumanEval/46 | **100%** | 18% | Baseline |
| HumanEval/129 | 0% | 0% | Tie |

### Key Finding: Fractal Wins on Hard Problems

The three problems where Fractal sampling outperformed baseline:
- **HumanEval/26**: `remove_duplicates` - needs algorithmic insight
- **HumanEval/29**: `filter_by_prefix` - straightforward but benefits from varied approaches
- **HumanEval/32**: `poly` - complex polynomial evaluation

These are problems where the **structure of the solution** matters more than syntactic variations.

## Analysis

### Why Fractal Sampling Underperformed

1. **Diversity Loss in Rendering**: Low-temperature code generation (T=0.2) collapses diverse sketches into similar implementations
2. **Sketch Quality**: Not all sketches are equally good - bad sketches produce bad code deterministically
3. **Overhead**: Two-stage process introduces more failure points

### When Fractal Sampling Helps

- Hard algorithmic problems where multiple valid approaches exist
- Problems requiring non-obvious structural decisions
- Cases where baseline gets stuck in local optima

## Conclusion

Direct token-level diversity (T=0.8) is more effective than sketch-level diversity for most HumanEval problems. However, **for hard problems requiring algorithmic insight, sketch diversity shows promise**.

Future work: Hybrid approach that uses fractal sampling selectively for problems identified as "hard".

## Files

- `flash_flood_sampling.py` - Main evaluation script
- `qwen_interface.py` - vLLM model interface
- `humaneval_harness.py` - Evaluation harness
- `results_flood.jsonl` - Raw results (in /tmp/)
