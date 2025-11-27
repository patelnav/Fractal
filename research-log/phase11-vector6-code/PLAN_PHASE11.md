# Phase 11: Vector 6 (Code Verification) - Execution Plan

**Objective:** Train an energy-based verifier to predict **execution success** for generated code, leveraging vLLM for high-throughput generation.

**Key Learning from Phase 10:**
*   Phase 10 evaluation took ~1.5 hours with HuggingFace.
*   vLLM offers **30x speedup** (verified).
*   **Action:** Use vLLM for all data generation and evaluation steps in Phase 11.

---

## Stage 1: Environment & Data Setup
**Goal:** Prepare the infrastructure for massive code generation and execution.

1.  **Model Selection:**
    *   Generator: `deepseek-ai/deepseek-coder-1.3b-instruct` (Fast, good for code).
    *   Verifier: `OuroborosModel` (Same architecture as Phase 7, but trained on code).
2.  **vLLM Setup:**
    *   Ensure vLLM is installed on Lambda (already done in Phase 10 test).
    *   Create `generator_vllm.py` adapted for DeepSeek (check chat templates).
3.  **Dataset:**
    *   Use **MBPP** (Mostly Basic Python Problems) via HuggingFace `datasets`.
    *   Sanitized version (hand-verified).
    *   Split: Train (374 problems), Test (500 problems) - *Check actual split sizes*.

## Stage 2: The "Infinite" Data Generator
**Goal:** Create 50,000+ `(Prompt, Code, Label)` triplets.

1.  **Generation Script (`generate_data.py`):**
    *   Use `VLLMGenerator`.
    *   For each problem in MBPP Train:
        *   Generate **100 candidates** (Temperature 1.0).
        *   Prompt format: `deepseek` chat template.
    *   Total volume: ~374 * 100 = 37,400 samples.
2.  **Execution Sandbox (`executor.py`):**
    *   Inputs: Generated code + MBPP test cases.
    *   Mechanism: `multiprocessing` with `timeout=2s` to run `exec()`.
    *   **Safety:** Run inside a minimal container or highly restricted user (Lambda is ephemeral, so acceptable risk if careful).
    *   Output: `PASS` (all tests passed) or `FAIL` (error/assertion fail).
3.  **Artifact:** `data_synthetic_code.jsonl`

## Stage 3: Training the Verifier
**Goal:** Train Ouroboros to assign Low Energy to Passing code and High Energy to Failing code.

1.  **Data Processing:**
    *   Input: `[CLS] Prompt [SEP] Code`
    *   Positive: `PASS` examples (Energy -> 0).
    *   Negative: `FAIL` examples (Energy -> 1).
    *   *Strategy:* For each problem, pair the (rare) passing solutions with randomly sampled failing solutions.
2.  **Training (`train_verifier.py`):**
    *   Loss: Contrastive Energy Loss (Margin Ranking).
    *   Batch Size: 64 (or max fit).
    *   Steps: ~10 epochs.

## Stage 4: Evaluation (The Proof)
**Goal:** Verify if the Energy Model improves Pass@1 on the Test Set.

1.  **Generate Candidates (Test Set):**
    *   Use vLLM.
    *   Generate 16 candidates per problem for all 500 test problems.
2.  **Reranking:**
    *   Score all 16 candidates with the Verifier.
    *   Select Candidate with Min(Energy).
3.  **Execution:**
    *   Run the Selected Candidate.
4.  **Metrics:**
    *   **Baseline Pass@1:** Random candidate accuracy.
    *   **Oracle Pass@16:** Best candidate accuracy (Upper Bound).
    *   **Verifier Pass@1:** Ouroboros accuracy.

## Timeline & Compute (Estimates)
*   **Generation:** 40k samples / 128 Hz (vLLM) ≈ 5 minutes on A100.
*   **Execution:** 40k samples * 0.1s ≈ 1 hour (CPU bound, parallelize).
*   **Training:** 1 hour.
*   **Evaluation:** 10 minutes.

**Total Lambda Time:** ~3-4 hours.
