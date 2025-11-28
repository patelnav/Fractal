# Phase 24: Fractal Critic - Solution Selection Evaluation

## Hypothesis
**"A Critic that analyzes the Step-by-Step Logic (Fractal analysis) of a solution is more accurate than a Critic that just gives a direct score."**

## Method

### Setup
- **Pool**: Generated 10 diverse code solutions per problem (T=0.8)
- **Ground Truth**: Executed each solution to determine pass/fail
- **Scoring**: Used Qwen2.5-Coder-7B-Instruct to score each solution with two methods

### Critic Strategies
1. **Baseline Critic**: Direct score (0-100) without analysis
2. **Fractal Critic**: Step-by-step analysis → then provide a score

### Metric
- **Best@1 Selection Accuracy**: Does the highest-scored solution actually pass the tests?

## Results

### Overall Selection Accuracy

| Method | Correct Selections | Total (with correct in pool) | Accuracy |
|--------|-------------------|------------------------------|----------|
| Baseline Critic | 10 | 12 | **83.3%** |
| Fractal Critic | 9 | 12 | **75.0%** |

**HYPOTHESIS NOT SUPPORTED**: Baseline Critic performed slightly better than Fractal Critic.

### Per-Problem Breakdown

| Task | Pool Has Correct? | Baseline Selected Correct | Fractal Selected Correct | Notes |
|------|-------------------|--------------------------|-------------------------|-------|
| HumanEval/0 | Yes | Yes | Yes | Tie |
| HumanEval/1 | Yes | Yes | Yes | Tie |
| HumanEval/10 | **No** | N/A | N/A | No correct solutions generated |
| HumanEval/11 | Yes | Yes | Yes | Tie |
| HumanEval/12 | Yes | Yes | Yes | Tie |
| HumanEval/26 | Yes | **No** | **No** | **CRITICAL: Both failed** |
| HumanEval/29 | Yes | Yes | Yes | Tie |
| HumanEval/32 | **No** | N/A | N/A | No correct solutions generated |
| HumanEval/33 | Yes | Yes | Yes | Tie |
| HumanEval/37 | Yes | Yes | Yes | Tie |
| HumanEval/39 | Yes | Yes | Yes | Tie |
| HumanEval/40 | Yes | Yes | **No** | **Baseline wins** |
| HumanEval/43 | Yes | Yes | Yes | Tie |
| HumanEval/46 | Yes | Yes | Yes | Tie |
| HumanEval/129 | **No** | N/A | N/A | No correct solutions generated |

### Critical Failure Analysis: HumanEval/26

**The `remove_duplicates` Problem**

Docstring: *"From a list of integers, remove all elements that occur more than once."*

Example: `remove_duplicates([1, 2, 3, 2, 4])` should return `[1, 3, 4]`

**Wrong Implementation (scored 100 by BOTH critics):**
```python
seen = set()
result = []
for num in numbers:
    if num not in seen:
        seen.add(num)
        result.append(num)
return result  # Returns [1, 2, 3, 4] - WRONG!
```

**Correct Implementation (scored 85 by baseline):**
```python
from collections import Counter
count = Counter(numbers)
return [num for num in numbers if count[num] == 1]  # Returns [1, 3, 4] - CORRECT!
```

**Analysis**: Both critics failed to distinguish between:
- "Remove duplicates" (keep first occurrence of each)
- "Remove elements that appear more than once" (keep only unique elements)

This reveals a fundamental limitation: **LLM critics struggle with subtle semantic distinctions** in problem requirements, regardless of whether they analyze step-by-step.

## Key Findings

1. **Step-by-step analysis doesn't improve selection accuracy** - The Fractal Critic actually performed slightly worse (75% vs 83.3%)

2. **Both critics fail on the same hard cases** - When the problem requires understanding subtle semantic requirements (HumanEval/26), both approaches fail equally

3. **The limitation is understanding, not methodology** - Adding "analyze step-by-step" doesn't help if the model fundamentally misunderstands the problem requirements

4. **Pool quality matters more** - 3 out of 15 tasks (20%) had NO correct solutions in the pool of 10. The critic can't select what doesn't exist

## Conclusions

The Fractal Critic hypothesis is **INCONCLUSIVE/NEGATIVE**:
- Step-by-step trace analysis does NOT improve solution selection over direct scoring
- Both approaches have similar blind spots
- The bottleneck is semantic understanding of requirements, not scoring methodology

**Recommendation**: Future work should focus on:
- Execution-based verification (unit tests)
- Multi-stage validation (generate test cases, then execute)
- Ensemble methods combining multiple critics

## Files
- `critic_eval.py` - Main evaluation script
- `qwen_interface.py` - vLLM model interface with `score_solution()` method
- `results_critic.jsonl` - Raw per-problem results
