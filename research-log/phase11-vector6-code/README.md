# Phase 11: Vector 6 - Program Synthesis & Execution Loop

**Objective:** Train an energy-based verifier to predict **execution success** for generated code.

## The Core Hypothesis
In Phase 10 (Math), the verifier failed because "reasoning correctness" is a soft, fuzzy target in the latent space. "Hallucination" often looks very similar to "Fact."

In **Vector 6**, we introduce **Hard Ground Truth**:
*   **Prompt:** "Write a function to sort a list."
*   **Candidate A:** `return list.sort()` (Correct-ish, but returns None. **FAIL**)
*   **Candidate B:** `return sorted(list)` (Correct. **PASS**)

To a language model, Candidate A and B look almost identical. To a compiler/interpreter, they are binary opposites.
**Hypothesis:** By training the verifier on `(Prompt, Code, Execution_Result)` triplets, we force the energy manifold to respect **functional correctness** rather than just **textual plausibility**.

## The Loop (The "Neuro-Symbolic" Cycle)
1.  **Generate:** LLM creates N candidate solutions for a coding problem.
2.  **Execute:** Run candidates against a test suite (sandbox).
3.  **Label:**
    *   Pass -> `Energy = 0`
    *   Fail -> `Energy = 1`
    *   Syntax Error -> `Energy = 1` (or higher)
4.  **Train:** Update Verifier to predict this binary outcome.
5.  **Inference:** Generate candidates -> Verifier Reranking -> (Optional) Execution.

## Roadmap

### Phase 11.1: Dataset Generation (The "Infinite" Source)
We need a massive dataset of `(Prompt, Code, Pass/Fail)` triplets.
*   **Source:** MBPP (Mostly Basic Python Problems).
*   **Tooling:** **vLLM** for generation (30x speedup confirmed in Phase 10).
*   **Action:**
    1.  Take the training set.
    2.  Generate 100 solutions per problem using `deepseek-ai/deepseek-coder-1.3b-instruct`.
    3.  Execute all 100.
    4.  Save the results. This creates a dataset of ~50k-100k examples with perfect labels.

### Phase 11.2: Verifier Training
Train the Ouroboros Verifier (energy head) on this dataset.
*   **Input:** `[CLS] Prompt [SEP] Code`
*   **Loss:** Contrastive Energy Loss (Push Pass down, Push Fail up).

### Phase 11.3: Evaluation
Test on the held-out Test Set.
*   **Baseline:** Pass@1 (Random sample from generator).
*   **Oracle:** Pass@N (If any candidate passes).
*   **Ouroboros:** Pass@1 (Selected by lowest energy).

## Why This Will Work (Where Phase 10 Failed)
1.  **Objective Signal:** No more "guessing" if the reasoning is right. The test suite is the oracle.
2.  **Hard Negatives:** We naturally generate "hard negatives" (bugs) that look like code but fail. These are the *perfect* training data for the verifier.
3.  **Verification is Easier than Generation:** Checking if code looks buggy is often easier than writing it from scratch.

## Resources
*   **Generator:** `deepseek-ai/deepseek-coder-1.3b-instruct` running on **vLLM**.
*   **Data:** MBPP (Sanitized) - easier to run than HumanEval.
*   **Compute:** Single A100 is sufficient for generation and training.
