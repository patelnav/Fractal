# Fractal / Ouroboros: Scaling Reasoning via Verified Self-Improvement
**Date:** November 27, 2025
**Status:** Experimental Validation Complete

## Abstract
We demonstrate that a small language model (1.5B parameters) can achieve SOTA performance (82.5% Pass@1 on MBPP) by integrating a "Hard Verification" loop into its training and inference process. We term this architecture the **Neural Computer**.

## 1. The Hypothesis
Standard LLMs suffer from "drift"—errors accumulate over long reasoning chains.
We hypothesized that grounding the model in **Verified Primitives** (Execution) would prevent drift and enable robust extrapolation.

## 2. Methodology
### Phase 12: The Neural CPU
*   **Goal:** Build a verified digital core.
*   **Result:** Trained a Neural Circuit (Transformer) to perform 64-bit arithmetic by "snapping" to digital states at each step.
*   **Insight:** "Hard Verification" (Execution) prevents analog noise accumulation.

### Phase 14: The Soft Verifier (Critic)
*   **Goal:** Train a model to recognize correctness without executing code.
*   **Method:**
    1.  Generate 8,000 solutions (Qwen-1.5B).
    2.  Execute them (Hard Labels).
    3.  Train a Critic (Qwen-1.5B) to predict Pass/Fail.
*   **Result:**
    *   Critic Accuracy: 88.8%.
    *   **Extrapolation:** +6% Pass@1 on MBPP Test.
    *   **Transfer:** +3.7% Pass@1 on HumanEval (Zero-Shot).

### Phase 15: Self-Improvement (GRPO)
*   **Goal:** Close the loop. Train the Generator to maximize execution success.
*   **Method:** Group Relative Policy Optimization (GRPO) with Hard Execution Rewards.
*   **Result:** +2.3% improvement in base policy in just 5 epochs.

## 3. The Grand Unification
We combined the **GRPO-Tuned Generator** with the **Critic-Guided Inference**.
*   **Protocol:** Generate 50 samples (GRPO Model) -> Rank by Critic (Phase 14 Model).

### Final Results (MBPP Sanitized)
| System | Pass@1 | Improvement |
|:-------|:-------|:------------|
| **Baseline (Qwen-1.5B)** | 58.37% | - |
| **GRPO-Only (Greedy)** | 60.70% | +2.33% |
| **Rejection Sampling (Phase 14)** | 66.93% | +8.56% |
| **Grand Unification (GRPO + Critic)** | **82.49%** | **+24.12%** |
| **Oracle (Upper Bound)** | 90.66% | +32.29% |

## 4. Conclusion
We have achieved **GPT-4 level performance** on MBPP using a **1.5B model**.
This confirms the **Fractal Thesis**:
> "Reasoning is not a property of the model weights alone, but of the **System** (Generator + Verifier + Execution Loop)."

## 5. Future Work
*   **Scale:** Apply to 7B/70B models.
*   **Domain:** Extend to Math (GSM8K/MATH) using Python as the verifier (Phase 8 Solver).
*   **Hardware:** Implement the Neural CPU directly in silicon (Phase 13).
