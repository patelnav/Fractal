# Phase 10 Results: Full System Evaluation

**Date:** 2025-11-26
**Model:** Gemma-3-1B-IT (Generator) + Ouroboros-Hardened (Verifier)
**Test Set:** GSM8K Test (1,319 problems)

## Summary Metrics

| Metric | Score | Count | Description |
| :--- | :--- | :--- | :--- |
| **Baseline (Pass@1)** | **52.46%** | 692/1319 | Random selection from generator (System 1) |
| **Oracle (Pass@16)** | **80.14%** | 1057/1319 | Best candidate exists in pool (Potential) |
| **Ouroboros (Pass@1)** | **50.42%** | 665/1319 | Verifier selection (System 2) |

## Analysis

### 1. The "Alignment Gap"
The generator is actually quite capable (80% potential accuracy), but the verifier fails to bridge the gap.
*   **Potential Gain:** +27.68% (Difference between Oracle and Baseline)
*   **Actual Gain:** -2.04% (Verifier is *worse* than random)

### 2. Verifier Pathology: The "Energy Sink"
The verifier assigns extremely low energy (high confidence) to obvious failures, indicating "holes" in the energy landscape.

**Example Failure 1: Repetition Loops**
*   **Problem:** "A robe takes 2 bolts..."
*   **Selected Answer:** "We are given that at the end... We are given that at the end..." (infinite loop)
*   **Energy:** 0.032 (Very Low / "Good")
*   **Diagnosis:** The verifier has not been hardened against repetition. It likely views these high-probability token sequences as "stable" states.

**Example Failure 2: Confident Hallucination**
*   **Problem:** House flipping profit calculation.
*   **Selected Answer:** "150% of $80,000 = 0.15 * $80,000 = $12,000" (Math Error)
*   **Energy:** 0.0019 (Extremely Low / "Good")
*   **Diagnosis:** The verifier cannot distinguish valid arithmetic from plausible-looking but wrong arithmetic.

### 3. Conclusion
Phase 10 is a **failure** in terms of performance improvement, but a **success** in revealing the limitations of the current Ouroboros implementation.
*   **The Hardening (Phase 9) was insufficient.** It likely didn't include enough "degenerate" negatives (like repetition loops) in the training mix.
*   **Energy-based models are fragile.** Without explicit constraints, they collapse into degenerate minima (repetition, simple patterns).

## Next Steps (Pivot)

We need to move away from "blind" energy minimization and toward **Structured Verification** or **Rule-Based Constraints**.

1.  **Filter Invalid Formats:** Hard-filter any candidate with repetition loops or missing `####`.
2.  **Math Consistency Check:** Use a Python sandbox to verify arithmetic steps (if possible).
3.  **Retrain Verifier:** We need a new training strategy. The current "Contrastive Divergence with valid/invalid pairs" isn't creating a steep enough barrier around correct reasoning.

**Recommendation:**
Abandon current Ouroboros checkpoint. Investigate **"Process Reward Models" (PRMs)** or **"Self-Correction"** loops where the generator critiques its own output, rather than a separate opaque energy model.
