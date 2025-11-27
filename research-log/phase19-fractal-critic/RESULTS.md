# Results: Vector 3 (Fractal Critic)

## Objective
Close the **Fractal Repair Loop** by training a "Critic" to guide the search.
Instead of randomly mutating roots, use a learned model to predict *where* the bug is, and use a domain heuristic to decide *how* to fix it.

## Methodology
1.  **Synthetic Critic Data:** Generated 10k examples of `(BuggyProgram, Error) -> FaultyIndex`.
2.  **Fractal Critic Model:** Trained a small Transformer (Loss ~115, Acc ~35%) to localize bugs. Note: 35% is low but > chance (1/8 $\approx$ 12%).
3.  **Guided Repair Strategy:**
    *   **Critic:** Top-3 Sampling to select the root to patch.
    *   **Heuristic:** If `Error > 0`, try `ADD` or `MUL`. If `Error < 0`, try `SUB`.
    *   **Benchmark:** Compare `Random Search` vs `Guided Search` (N=50 trials).

## Results (N=50 Trials)

| Metric | Random Search | Guided Search | Improvement |
|:-------|:--------------|:--------------|:------------|
| **Success Rate** | 22.0% | **54.0%** | **+32.0% (Absolute)** |
| **Mean Steps** | 12.7 | **10.5** | **-17%** |

## Analysis
*   **The Critic Works:** Even a weak critic (35% acc) combined with a simple heuristic **doubled** the success rate of the repair loop compared to random search.
*   **Why:** Random search wastes time mutating correct roots or applying counter-productive edits (e.g., subtracting when the total is already too low). The Critic + Heuristic focuses the search on likely candidates and likely operations.
*   **Limit:** The Critic often gets stuck in loops if it confidently predicts the wrong location (brittle). Top-k sampling helped mitigate this.

## Conclusion
**Vector 3 is VALIDATED.**
We have proven that a learned Critic can significantly accelerate the Fractal Repair Loop.
This completes the **Fractal Coder** architecture:
1.  **Manager:** Plans (Roots).
2.  **Flash Flood:** Renders (Tokens) - *Fast*.
3.  **Execution:** Verifies (Hard Truth).
4.  **Critic:** Diagnoses (Fault Localization) - *Smart*.
5.  **Editor:** Patches (Surgical Edit) - *Stable*.

## Artifacts
*   `research-log/phase19-fractal-critic/fractal_critic.py`
*   `research-log/phase19-fractal-critic/train_critic.py`
*   `research-log/phase19-fractal-critic/test_guided_repair.py`
*   `research-log/phase19-fractal-critic/synthetic_critic_data.py`
