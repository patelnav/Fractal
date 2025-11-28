# Phase 24: Fractal Critic

## Objective
Solve the **Selection Problem**: We have generated diverse solutions (Flash Flood), but we don't know which one is correct without an Oracle.
We will build a **Critic** to rank the generated solutions.

## Hypothesis
"A Critic that analyzes the **Step-by-Step Logic** (Reverse-Sketching) of a solution is more accurate than a Critic that just looks at the Code."

## Architecture

### 1. The Pool
Reuse the `results_flood.jsonl` from Phase 23. We have 50 Baseline + 50 Fractal solutions per problem.
Total Pool: ~100 candidates.

### 2. The Critic Strategies
Compare three Critic approaches:
- **A. Baseline Critic:** "Rate this code from 0-10 on correctness."
- **B. Test-Case Critic:** "Generate a test case, run it, and check result." (Standard Python execution).
- **C. Fractal Critic (Trace Analysis):**
    1.  **Reverse Sketch:** "Explain this code step-by-step."
    2.  **Logical Check:** "Does this explanation match the Docstring requirements? Y/N."

## Experiment
1.  **Load Pool:** For each of the 15 problems, load the 100 generated solutions.
2.  **Label:** We know the ground truth (Passed/Failed) from Phase 23.
3.  **Run Critics:** Ask each Critic to score each solution.
4.  **Evaluate:** Calculate **AUROC** (Area Under ROC Curve) or **Best@1** (Probability that the top-ranked solution is correct).

## Success Metric
- Can the Fractal Critic pick a correct solution from the pool more reliably than the Baseline Critic?
- If `Best@1 (Critic)` approaches `Pass@100 (Oracle)`, we have a solved system.

## Tasks
- [ ] `critic_analysis.py`: Script to load results and run scoring.
- [ ] Implement the prompts.
- [ ] Run evaluation.
