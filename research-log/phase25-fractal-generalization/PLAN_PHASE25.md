# Phase 25: Fractal Generalization Benchmark

## Objective
Demonstrate a fundamental **Intelligence Jump** where a Fractal (weight-shared, recursive) architecture generalizes to problem depths unseen during training, while a standard Transformer baseline fails.

## The Domain: "Stateful Dyck-N"
We need a task that requires recursive logic + state tracking.
**Task:** Evaluate a nested expression where each nesting level modifies a value.
**Format:** `(Op Value (Op Value ... ) ) = Result`
**Rules:**
- `Op` is `+` or `*`.
- `Value` is an integer (0-9).
- Nesting Depth determines operation order (Standard math, or a specific recursive rule).
- **Crucially:** The "Rule" is invariant across levels.
- **Example:** `(+ 2 (* 3 (+ 1 1)))` -> `(+ 2 (* 3 2))` -> `(+ 2 6)` -> `8`.

This forces the model to learn the *process* of evaluation (the recursive rule), not just memorizing patterns of tokens.

## Experiment Design

### 1. Models (Parameter Matched ~5M)
- **Baseline (GPT-Standard):**
    - 6 Layers, 4 Heads, 128 Emb.
    - Standard Positional Encoding.
    - Trained AR (Next Token).
- **Fractal (Recursive-GPT):**
    - 1 Layer Block (Shared).
    - Applied $D$ times (where $D$ matches the depth of the input, or fixed high number).
    - Or: Latent Recurrence?
    - **Simpler Approach:** Discrete Diffusion / Refinement.
        - Input: `Masked Sequence`.
        - Step 1: Predict Level 1 structure.
        - Step 2: Refine Level 2.
        - All steps use the **SAME weights**.

**Wait, let's stick to the simpler Recurrent-depth definition for direct comparison.**
FractalGPT = A Transformer where layers $1..N$ share the *same weights*.
It is literally a 1-layer transformer unrolled $N$ times.
Baseline = A Transformer with $N$ *distinct* layers.
If Fractal beats Baseline on OOD Depth, it proves **Weight Sharing induces Algorithmic Generalization**.

### 2. Data
- **Train:** Depths 1-4.
- **Test:** Depths 5-12.
- **Vocab:** `( ) + * 0 1 2 3 4 5 6 7 8 9 =`

### 3. Training
- Train both to convergence on Depths 1-4.
- Use standard Cross-Entropy Loss.

### 4. Evaluation
- Metric: Exact Match Accuracy on the final result (after `=`).
- Plot: Accuracy vs Depth (1..12).

## Implementation
- `fractal_data_gen.py`: Generates the expressions and solutions.
- `models.py`: Implements `NanoGPT` and `FractalGPT` (Shared Weights).
- `train_compare.py`: Training loop for both.
- `evaluate_generalization.py`: Sweep over depths.

## Expected Outcome
- **Depths 1-4:** Both models ~100%.
- **Depths 5-12:**
    - Baseline: Degrades rapidly (overfitted to training depth).
    - Fractal: Maintains high accuracy (applied the learned rule recursively).
