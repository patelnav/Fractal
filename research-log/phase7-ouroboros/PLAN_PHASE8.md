# Phase 8: Self-Correcting Solver

## Context

Phase 7 achieved **80% validation detection rate** on reasoning chains (GSM8K + HumanEval). The 20% train/val gap indicates the model learned perturbation heuristics rather than true logical consistency.

**Key Insight**: The model is a useful *filter* even if imperfect. We can use it to rank multiple candidate solutions.

## Objective

Prove that Ouroboros energy-based verification boosts reasoning accuracy by filtering N candidate solutions and selecting the one with lowest energy.

## Hypothesis

Given N=5 candidate solutions to a math problem:
- **Baseline**: First candidate (greedy decode)
- **Ouroboros**: Candidate with minimum energy

If Ouroboros accuracy > Baseline accuracy, energy-based verification works for reasoning.

## Experiment Design

### Components
1. **Generator**: Small LM to produce candidate solutions (GPT-2 or TinyLlama)
2. **Verifier**: Phase 7 Ouroboros checkpoint (`checkpoints/ckpt.pt`)
3. **Benchmark**: GSM8K test set (1,319 problems)

### Evaluation Loop
```python
for problem in gsm8k_test:
    candidates = generator.generate(problem, n=5)
    energies = [ouroboros.get_energy(problem, c) for c in candidates]

    baseline_correct = is_correct(candidates[0])
    ouroboros_correct = is_correct(candidates[argmin(energies)])
    oracle_correct = any(is_correct(c) for c in candidates)
```

### Metrics
| Metric | Description |
|--------|-------------|
| Baseline Accuracy | % correct using first candidate |
| Ouroboros Accuracy | % correct using min-energy candidate |
| Oracle Accuracy | % where at least one candidate correct |
| Lift | (Ouroboros - Baseline) / Baseline |

## Implementation

### Files to Create
1. `solve_math.py` - Main evaluation script
2. `generator.py` - Wrapper for candidate generation
3. `config/eval_gsm8k.py` - Evaluation configuration

### Expected Results
```
GSM8K Evaluation (N=5 candidates)
==================================
Baseline (Greedy):      ~40%
Ouroboros (Min-Energy): ~55-60%  (+40% lift)
Oracle (Any Correct):   ~70%

Energy Statistics:
  Correct candidates: mean=0.25, std=0.15
  Wrong candidates:   mean=0.65, std=0.20
```

## Stretch Goal: Adversarial Retraining

After Phase 8, use failure cases to improve Ouroboros:

1. **Hard Negatives**: Candidates with low energy but incorrect answers
2. **Retrain**: Add these to training set as wrong examples
3. **Iterate**: Repeat until val detection > 90%

This creates a self-improving loop where the verifier gets harder negatives from its own failures.

## Success Criteria

- [ ] Ouroboros accuracy > Baseline accuracy (proves verification works)
- [ ] Energy correlates with correctness (lower = more likely correct)
- [ ] Identify failure modes for future improvement

## Resources

- Phase 7 checkpoint: `checkpoints/ckpt.pt` (764MB, 63.6M params)
- GSM8K test: `data/gsm8k/test.json` (1,319 problems)
- Compute: Local M2 sufficient for evaluation (no training needed)
