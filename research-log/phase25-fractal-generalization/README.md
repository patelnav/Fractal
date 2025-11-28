# Phase 25: Fractal Generalization (Dyck-N)

## Objective
Test if a **Fractal Transformer** (1 layer with shared weights, looped $N$ times) can generalize to deeper recursive structures than a **Baseline Transformer** (fixed $N$ layers) by dynamically increasing the loop count at test time.

## Experiment
- **Task:** Dyck-N Bracket Matching (Dense).
    - Input: `[ { ( <`
    - Output: `> ) } ]`
- **Training:** Depths 1-6.
- **Testing:** Depths 1-14.
- **Models:**
    - Baseline: 6-Layer GPT (Standard).
    - Fractal: 1-Layer GPT (Shared), trained with 6 loops.

## Results

| Depth | Baseline | Fractal (Fixed Loops=6) | Fractal (Dynamic Loops=Depth) |
|-------|----------|-------------------------|-------------------------------|
| 1-6   | **100%** | **100%**                | **100%**                      |
| 7-14  | 0%       | 0%                      | 0%                            |

## Conclusion: Negative
The Fractal Architecture **did not** enable length generalization.
1.  **In-Distribution:** Both models solved the task perfectly.
2.  **Out-of-Distribution:** Both models failed completely.
3.  **Dynamic Looping:** Increasing the loop count for deeper problems did not help.

## Analysis
The Dyck-N task requires **Stack Memory** (remembering the sequence of open brackets), not just **Iterative Computation**.
Looping the weights allows the model to "think longer," but it does not expand its memory capacity or change its attention span. The model likely overfitted to absolute positions seen during training (Pos 1-12).

## Next Steps
Pivot to **Phase 26: Flash Flood at Scale**.
Instead of proving "Smarter Structure," we will demonstrate **"Faster Structure"**.
We will benchmark the `Flash Flood` parallel decoding algorithm against standard autoregressive generation on a large-scale structured task.
