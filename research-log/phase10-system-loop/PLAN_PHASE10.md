# Phase 10: Full System Loop (Evaluation)

**Objective:** Quantify the performance impact of the **Adversarial Hardening (Phase 9)** on the unseen **GSM8K Test Set**.

**Context:**
Initial attempts (Phase 10.0) failed due to:
1.  **Generator Failure:** Gemma-3-1B-IT produced repetitive garbage loops and leaked chat template tags when manually prompted.
2.  **Verifier Crash:** Ouroboros (trained on 512 max len) crashed on inputs > 512 tokens due to RoPE shape mismatch.
3.  **Baseline Collapse:** Baseline accuracy was ~4-12% (expected ~30%), indicating extraction/generation pipeline breakage.

---

## The Revised Plan (Phase 10.1)

We will fix the pipeline to ensure stable generation and valid verification.

### 1. Fix Generation & Extraction
*   **Generator:** Revert to `tokenizer.apply_chat_template(..., tokenize=True)` which handles special tokens correctly.
*   **Prompting:** Ensure `add_generation_prompt=True` is used so the model knows it's the assistant's turn.
*   **Extraction:** Improve regex to handle `#### \n Answer` or `####` at the very end of string.

### 2. Strict Constraints
*   **Length Limit:** Enforce `len(context) + len(target) <= 512`.
    *   Action: Truncate Context (Question) to 256.
    *   Action: Truncate Target (Answer) to 256.
    *   Reason: Ouroboros RoPE embeddings are fixed at 512. Extrapolation causes crashes.

### 3. Evaluation Steps
1.  **Sanity Check (N=32):** Run a small batch to verify Baseline Accuracy is > 25%.
    *   If Baseline < 25%, **STOP**. Do not verify. Fix Generator.
2.  **Full Evaluation (N=1319):** Once baseline is healthy, run full test.
    *   Generate 16 candidates.
    *   Verify with Hardened Ouroboros.
    *   Compare Baseline (Pass@1) vs Ouroboros (Verifier).

### 4. Success Metrics

| Metric | Baseline Target | Ouroboros Target |
|--------|-----------------|------------------|
| **Accuracy** | > 30% | > 45% |

## Execution

Script: `run_eval.py`
*   Update to use `HuggingFaceGenerator` with corrected template logic.
*   Implement strict tokenizer-based truncation.
*   Add `LIMIT` flag for rapid debugging.