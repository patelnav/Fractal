# Phase 10: Full System Loop (Evaluation)

**Objective:** Quantify the performance impact of the **Adversarial Hardening (Phase 9)** on the unseen **GSM8K Test Set**.

**Context:**
Initial attempts (Phase 10.0) failed due to generator formatting issues and verifier shape mismatches. We are adopting a strict, gated testing protocol to prevent wasted compute.

---

## The Gated Execution Plan (Phase 10.1)

We will proceed in three distinct stages. **Do not proceed to the next stage unless the current stage passes.**

### Stage 1: Generator & Pipeline Qualification
**Goal:** Ensure Gemma-1B generates valid, parseable answers and the extraction logic works.
*   **Action:** Run `test_generator.py` on 20 test problems.
*   **Checks:**
    1.  **Format:** Output must contain `#### [Answer]`.
    2.  **Cleanliness:** No leaked chat tags (`<start_of_turn>`, `user`, `model`) or repetition loops.
    3.  **Baseline Accuracy:** Must be > 25% on this small sample.
*   **Failure Condition:** If accuracy < 25% or format is broken, **STOP**. Fix `generator.py` or `utils.py`.

### Stage 2: Verifier Integration Test
**Goal:** Ensure Ouroboros runs without crashing and produces valid energy scores.
*   **Action:** Run `run_eval.py` with `LIMIT=32` and `BATCH_SIZE=8`.
*   **Constraints:**
    *   Enforce strict length limit: `len(context) + len(target) <= 512` (Truncate inputs to 256/256).
*   **Checks:**
    1.  **Stability:** No `RuntimeError: shape invalid` (RoPE errors).
    2.  **Scoring:** Energy scores are in [0, 1] range and not all identical.
*   **Failure Condition:** If crashes occur, **STOP**. Fix truncation logic in `run_eval.py`.

### Stage 3: Full Evaluation
**Goal:** Measure final system performance.
*   **Action:** Run `run_eval.py` on full 1,319 test set.
*   **Config:** `N_CANDIDATES=16`, `BATCH_SIZE=8`.
*   **Success Metrics:**
    *   **Baseline (Pass@1):** > 30%
    *   **Ouroboros (Verifier):** > 45% (Hypothesis: Hardening adds +15% over baseline)

---

## Implementation Details

### 1. Strict Truncation Strategy
Since Ouroboros was trained with `max_seq_len=512`, we cannot feed it longer sequences.
*   **Context (Question):** Truncate to 256 tokens.
*   **Target (Answer):** Truncate to 256 tokens.
*   **Total:** 512 tokens max.

### 2. Generator Prompting
Revert to standard `apply_chat_template` to fix formatting issues, but monitor output closely in Stage 1.

### 3. Artifacts
*   `results_stage1_gen.jsonl`: Generator debug outputs.
*   `results_stage2_ver.jsonl`: Verifier debug outputs.
*   `results_phase10.jsonl`: Final evaluation.
