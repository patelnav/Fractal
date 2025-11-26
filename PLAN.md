# Project: The Fractal Falsifiable Diffusion Engine
**Based on:** *Talagrand’s Convolution Conjecture via Perturbed Reverse Heat* (Yuansi Chen, 2025)
**Objective:** Validate a hyper-sample-efficient, recursive architecture for discrete generative AI.

---

## 1. The Theoretical Breakthrough
We started with a dense mathematical paper by Yuansi Chen.
*   **The Problem:** Standard "Diffusion Models" (like Stable Diffusion) rely on Gaussian noise and continuous calculus. They don't work natively on "Discrete" data like Text, DNA, or Code ($\{-1, 1\}^n$).
*   **The Solution:** The paper proves that the **Heat Semigroup** on the Boolean Hypercube smooths out discrete functions in a way that creates "Positive Curvature" (Semi-Log-Convexity).
*   **The Implication:** It gives us a mathematically rigorous way to define **"Scores"** (gradients) on bits. It proves that a **Reverse Heat Process** (denoising) can recover the original data distribution from pure Poisson noise.
*   **Why it matters:** This allows us to build a Diffusion model for text that doesn't rely on "rounding" or "embeddings hacks." It is native, mathematically bounded, and inherently stable.

## 2. The Architectural Evolution
Our discussion moved from a basic implementation to a novel theoretical architecture.

### Phase A: The "Parallel" Shift
We realized that by replacing Autoregressive (Next-Token) Logic with **Discrete Diffusion**, we gain:
*   **Global Coherence:** The model sees the end of the sentence while generating the start.
*   **Parallelism:** We can generate 1,000 tokens in $O(1)$ time relative to length.
*   **Implementation:** We strip the Causal Mask from `nanoGPT` and introduce a Time-embedding MLP.

### Phase B: The "Epistemic" Shift (Popperian AI)
We integrated Karl Popper’s philosophy of Falsification.
*   **The Insight:** Chen’s **Discrete Score ($S_i$)** is not just a predictor; it is a measure of **Energy/Friction**.
*   **The Feature:** If a model tries to generate a "Lie" (e.g., QBasic doing HTTP requests), the diffusion process encounters high resistance in the hypercube. We can measure this energy to let the model self-detect hallucinations and say "This is impossible."

### Phase C: The "Fractal" Shift (The Holy Grail)
We concluded that to build a **Tiny, Sample-Efficient General Intelligence**, we must abandon specialized layers.
*   **The Hypothesis:** Intelligence is **Scale Invariant**. The logic required to expand a "Chapter" into "Paragraphs" is the same logic required to expand a "Sentence" into "Words."
*   **The Design:** A **Recursive Discrete Diffusion (RDD)** model.
    *   One tiny model.
    *   Shared weights.
    *   Called recursively to "zoom in" on data, refining coarse tokens into fine tokens.

---

## 3. The Critical Unknown (The Risk)
Before building the full system, we identified a massive theoretical risk: **Gradient Conflict.**

We are asking one set of neural weights to understand two different "Manifolds":
1.  The distribution of High-Level concepts (Roots $\to$ Chunks).
2.  The distribution of Low-Level details (Chunks $\to$ Fine Tokens).

If the **Discrete Diffusion Operator** is not robust enough, the gradients from these two tasks will cancel each other out, and the model will fail to converge. **We cannot proceed without ruling this out.**

---

## 4. The Execution Plan: The "Smallest Falsifiable Test"

We will run a specific, isolated experiment to prove or disprove the **Universal Refinement Hypothesis**.

### The Experiment: "The 1-to-4 Recursive Expansion"

#### A. The Synthetic Dataset (The Toy Universe)
We create a strictly hierarchical language to simulate "Abstraction."
*   **Level 0 (Root):** Integers `0-9`.
*   **Level 1 (Chunks):** A deterministic mapping (e.g., `5` $\to$ `[A, B, A, B]`).
*   **Level 2 (Fine):** A deterministic mapping (e.g., `A` $\to$ `[10, 11, 10, 11]`).

#### B. The Subject (The Model)
*   **Base:** `nanoGPT` (Forked).
*   **Size:** Microscopic (2 Layers, 4 Heads, 64 Embed Dim).
*   **Architecture:**
    *   **No Causal Mask** (Bidirectional Attention).
    *   **Time Embeddings** (Input $t$ into the transformer).
    *   **Input Signature:** `[Condition_Token] + [Noisy_Target_Tokens]`.

#### C. The Training Protocol
We train the *same model instance* on two tasks simultaneously:
1.  **Task A:** Denoise Level 1 tokens given a Level 0 Root.
2.  **Task B:** Denoise Level 2 tokens given a Level 1 Chunk.
*   *Note:* The weights are shared. The model must learn to switch context based solely on the input token.

#### D. The Test (Falsification)
After training, we run the **Reverse Heat Generation** recursively:
1.  Feed `Root(5)` $\to$ Generate 4 Tokens (Expect `[A, B, A, B]`).
2.  Feed the *generated* `A` $\to$ Generate 4 Tokens (Expect `[10, 11, 10, 11]`).

---

## 5. Success Criteria & Interpretations

### Outcome 1: Failure (The Model Hallucinates)
*   **Symptom:** The model learns Task A but fails Task B, or creates a "mush" of both.
*   **Meaning:** The **Universal Refinement Hypothesis is False** for this architecture. The Discrete Score Function is sensitive to the specific topology of the data manifold and cannot generalize across scales.
*   **Action:** We abandon the Fractal approach. We switch to a **Mixture of Experts (MoE)** where different layers handle different scales.

### Outcome 2: Success (Perfect Reconstruction)
*   **Symptom:** The model achieves >99% accuracy on the full recursive expansion.
*   **Meaning:** The **Universal Refinement Hypothesis is Viable**. We have proven that Chen’s math allows a model to learn the *General Law of Expansion* rather than just memorizing patterns.
*   **Implication:** We can proceed to build the **Fractal Language Model**—a tiny, incredibly smart model that beats larger models by recycling its own weights to "think" at different levels of abstraction.

## 6. Immediate Next Steps
1.  **Fork** `karpathy/nanochat` or `nanoGPT`.
2.  **Write** `fractal_data.py` to generate the 3-layer toy hierarchy.
3.  **Modify** `model.py` to remove the causal mask and add `time_mlp`.
4.  **Modify** `train.py` to implement the Poisson Noise / Score Matching loss.
5.  **Run** the training loop.

This test is the gateway. If it passes, we aren't just building a chatbot; we are building a recursive, self-verifying intelligence engine.