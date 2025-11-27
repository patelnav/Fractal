# Phase 14 Results: Vector 6 Reboot (Verified Code Generation)

**Date:** November 27, 2025
**Model:** Qwen2.5-Coder-1.5B-Instruct
**Hardware:** NVIDIA A100 (Lambda Labs)

## Executive Summary
We successfully validated the **Vector 6 Hypothesis**: "Hard Verification Drives Soft Extrapolation."
By generating massive samples ($N=50$), executing them (Hard Verification), and training a Critic on the results, we created a Soft Verifier that significantly improves Pass@1 via Rejection Sampling.

Crucially, this Verifier **transfers** to unseen benchmarks (HumanEval) zero-shot.

## 1. MBPP (In-Domain)
*   **Training Data:** ~8,000 samples (MBPP Train).
*   **Test Data:** ~12,800 samples (MBPP Test).
*   **Critic:** Qwen-1.5B (Epoch 3, Val Acc 88.8%).

| Metric | Value |
|:-------|:------|
| **Baseline Pass@1** (Random) | 60.94% |
| **Critic Pass@1** (Top-1) | **66.93%** |
| **Improvement** | **+5.98%** |
| **Oracle Pass@N** (Upper Bound) | 90.27% |

## 2. HumanEval (Out-of-Domain Transfer)
*   **Test Data:** ~8,000 samples (164 problems x 50).
*   **Critic:** Same MBPP-trained model (Zero-shot).

| Metric | Value |
|:-------|:------|
| **Baseline Pass@1** (Random) | 64.62% |
| **Critic Pass@1** (Top-1) | **68.29%** |
| **Improvement** | **+3.67%** |
| **Oracle Pass@N** (Upper Bound) | 95.73% |

## 3. Negative Result: Hardening
We attempted to "harden" the Critic by mining False Positives (High score, failed execution) and fine-tuning on them.
*   **Result:** Performance dropped (66.9% -> 65.7%).
*   **Interpretation:** Over-weighting confusing edge cases degraded the robust decision boundary learned from the broader distribution.

## Conclusion
The **Software** component of the Neural Computer is viable.
1.  **Generate** (Creative/Stochastic).
2.  **Execute** (Hard Truth).
3.  **Learn** (Train Critic on Execution).
4.  **Verify** (Use Critic to guide Generation).

This loop is closed and functional.
