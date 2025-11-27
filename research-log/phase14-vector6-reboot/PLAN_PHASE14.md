
# Phase 14: Vector 6 Reboot (Verified Code Generation)

**Objective:** Apply the "Hard Verification" insight from the Neural CPU (Phase 12) to Real-World Code Generation.

## The Thesis
Phase 12 proved that **Digital Restoration** (snapping to a verified state) prevents drift in neuro-symbolic systems.
In Code Generation, the "Verified State" is **Passing Unit Tests**.
We believe that training a model to optimize for *Test Passing* (Execution) rather than *Text Matching* (Perplexity) will yield robust reasoning capabilities.

## The Pipeline
1.  **Generator:** `Qwen/Qwen2.5-Coder-1.5B-Instruct` (State-of-the-art 1.5B model, ~59% MBPP Pass@1).
2.  **Executor (Hard Verifier):** Python Sandbox that runs generated code against MBPP unit tests.
3.  **Critic (Soft Verifier):** A model trained to predict `P(Pass | Prompt, Code)`.
    *   Input: `[PROMPT] + [CANDIDATE CODE]`.
    *   Objective: Binary Classification (Pass/Fail).

## The Experiment Loop (Ouroboros 2.0)
1.  **Generate (Data Creation):**
    *   For each MBPP Training problem, sample $N=50-100$ solutions using the Generator (temperature > 0).
    *   Execute all samples. Store `(Prompt, Code, Result)`.
    *   This creates a rich dataset of "Correct", "Near Miss", and "syntax error" examples.
2.  **Train Critic:**
    *   Train a BERT-like or Decoder-only verifier to distinguish Pass vs Fail.
    *   Curriculum: Start with balanced Pass/Fail. Later, mine "Hard Negatives" (failures that look correct).
3.  **Evaluate (Extrapolation):**
    *   **Baseline:** Generator Pass@1 on Held-out MBPP Test set.
    *   **Oracle:** Pass@k (If any of k samples pass).
    *   **Verifier:** Rank k samples using the Critic. Select Top-1.
    *   **Goal:** Verifier Pass@1 > Baseline Pass@1.

## Datasets
*   **MBPP:** Mostly Basic Python Problems (Primary).
    *   Train: Problems 1-500.
    *   Test: Problems 501-974.
*   **HumanEval:** (Secondary / Test).

## Infrastructure
*   **Lambda Labs A100:** For Generation & Training.
*   **Local/Remote Sandbox:** `exec` with timeout and memory limits.
