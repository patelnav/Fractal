# Phase 11: Fractal Logic Gates (The Differentiable ALU)

**Objective:** Train a discrete diffusion model to perform exact arithmetic operations by learning the underlying logic circuit on the Boolean Hypercube.

**The Insight:**
Standard Neural Networks (MLPs/Transformers) struggle with arithmetic because they try to *approximate* discrete logic using continuous functions. They memorize "2+2=4" but fail at "2342+5678".
Our technology (Chen's Discrete Diffusion) is mathematically proven to handle discrete transitions on the hypercube while remaining differentiable.

**Hypothesis:** A Fractal Diffusion model can act as a "Neural ALU," learning the generalized algorithm for Addition/Multiplication rather than memorizing the lookup table.

---

## Stage 1: The Binary Adder (Proof of Concept)
**Task:** Learn to add two N-bit integers.
*   **Input:** Two binary strings `A` and `B` (concatenated or embedded).
*   **Output:** Binary string `C = A + B`.
*   **Model:** Small Transformer trained with Fractal Diffusion (Phase 1 style).
*   **Test:**
    *   **In-Distribution:** Train on 8-bit numbers, test on held-out 8-bit numbers.
    *   **Extrapolation (The Holy Grail):** Train on 8-bit numbers, test on 12-bit numbers. (If it learns the *algorithm* of "carry the one", it should generalize).

## Stage 2: The Multiplier (Complexity Test)
**Task:** Learn to multiply.
*   Multiplication is harder because it involves $O(N^2)$ operations (shift and add) vs $O(N)$ for addition.
*   If the model can learn this, it effectively learns a complex circuit.

## Stage 3: Integration (The Co-Processor)
**Task:** Graft this module onto a pre-trained LLM (e.g., Gemma).
*   Instead of the LLM hallucinating the answer, it learns to "route" the number embeddings to the Fractal ALU.
*   The Fractal ALU "diffuses" the answer.
*   The result is fed back to the LLM.

---

## Execution Plan

### 1. Data Generation (`generate_binary_data.py`)
*   Generate pairs of integers (A, B) and sum C.
*   Convert to fixed-width binary strings.
*   Split: Train (Small numbers), Test (Large numbers).

### 2. Model Setup (`model_logic.py`)
*   Reuse `Phase 4` Fractal Engine architecture.
*   Modify Tokenizer: Vocabulary is just `0`, `1`, `PAD`, `START`.
*   Loss: Discrete Diffusion Loss (Chen's Energy).

### 3. Training (`train_logic.py`)
*   Train on 8-bit addition.
*   Monitor: Exact Match Accuracy.

### 4. Evaluation (`eval_extrapolation.py`)
*   Feed 12-bit numbers (never seen during training).
*   Measure if the model correctly calculates the sum.

## Success Criteria
*   **Pass:** >99% accuracy on held-out 8-bit sums.
*   **Breakthrough:** >90% accuracy on 10-bit sums (Extrapolation).
