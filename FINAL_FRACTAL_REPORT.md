# The Fractal Research Arc: Final Report

**Date:** November 27, 2025  
**Status:** Concluded  
**Principal Finding:** Negative Result for Causal Architectures; Strong Signal for Bidirectional Diffusion.

---

## Executive Summary

The **Fractal Project** explored three core hypotheses regarding the next generation of AI architectures:
1.  **Recursive Intelligence:** Can shared-weight transformers generalize to infinite depth?
2.  **Fractal Reasoning:** Can sketching and self-explanation stabilize generation and repair bugs?
3.  **Flash Flood Decoding:** Can parallel decoding achieve $O(1)$ latency?

**Conclusion:** While the system demonstrated massive raw throughput potential (**94x speedup**), the core hypothesis that "Causal LLMs can be prompted/looped to behave like Fractal Systems" was **disproven**. The fundamental constraint of Causal Masking prevents the information flow required for true parallel convergence. The future of high-speed, high-intelligence systems lies in **Bidirectional Diffusion Models**, not Causal Autoregression.

---

## Part 1: The Recursive Hypothesis (Infinite Depth)

**Hypothesis:** A shallow Transformer block, if looped $N$ times with shared weights (Fractal Architecture), should generalize to problem depths unseen during training by dynamically increasing $N$ at inference time.

**Experiments:**
-   **Phase 4-5:** Toy tasks showed promise with discrete diffusion steps.
-   **Phase 25 (Generalization Benchmark):** We trained a Shared-Weight GPT (1 layer looped 6 times) vs a Baseline GPT (6 layers) on recursive arithmetic and Dyck-N bracket matching (Depths 1-6). Tested on Depths 7-14.

**Results:**
-   **In-Distribution:** Both models achieved ~100% accuracy.
-   **Out-of-Distribution:** Both models failed completely (0% accuracy).
-   **Dynamic Looping:** Increasing the loop count at test time did *not* help the Fractal model.

**Key Finding:**
**Computation $\neq$ Memory.** Looping a block adds non-linear processing power, but it does not extend the "Working Memory" required to track deeper stacks. The model overfits to the positional embeddings of the training distribution regardless of recurrence.

---

## Part 2: The Reasoning Hypothesis (Self-Correction)

**Hypothesis:** A model that generates a high-level "Sketch" (Plan) before coding, or "Explains" code step-by-step (Fractal Critic), will outperform a baseline model in correctness and repair capability.

**Experiments:**
-   **Phase 21-22 (Repair Loop):** Attempted to fix "Hard" HumanEval bugs by refining the Plan based on error traces.
-   **Phase 24 (Fractal Critic):** Compared a standard "Rate 0-100" Critic vs a "Step-by-Step Analysis" Critic for selecting the best solution from a diverse pool.

**Results:**
-   **Repair:** 0% success rate on the "Hard 5" problems. The model would "thrash" (edit without fixing) because it lacked the fundamental algorithmic knowledge.
-   **Critic:** The Fractal Critic (75% acc) performed *worse* than the Baseline Critic (83% acc). Asking for an explanation introduced hallucinations that justified the bug.

**Key Finding:**
**Planning is not a Knowledge Substitute.** If a model doesn't know the algorithm (e.g., how to traverse a grid), asking it to "Plan" just produces a confident, wrong plan. Reasoning structure cannot create information that isn't in the weights.

---

## Part 3: The Speed Hypothesis (Flash Flood)

**Hypothesis:** We can generate text in $O(1)$ parallel steps (Flash Flood) by initializing with a "Sketch" and refining all tokens simultaneously, beating the $O(N)$ bottleneck of Autoregressive (AR) generation.

**Experiments:**
-   **Phase 26 (Scale):** Benchmarked raw throughput of 10 parallel passes vs AR.
-   **Phase 27 (Latency):** Tested "Sketch + Flood" pipeline.
-   **Phase 29 (Init Proof):** Tested "Jacobi Decoding" on Qwen-1.5B with correct "Islands" initialized.

**Results:**
-   **Speed:** Confirmed. **94x Speedup** on long sequences (2048 tokens generated in 0.5s vs 51s).
-   **Convergence (Phase 29):** **Failed.** Even when initialized with 50% correct tokens (Islands), the standard Causal Model destroyed them in Step 2, dropping accuracy from 22% to 6%.

**Key Finding:**
**Causal Masking Kills Parallelism.** In a Causal model, Token $i$ depends *only* on $0..i-1$. If Token $i-1$ is wrong (which it is during initialization), Token $i$ becomes garbage, overwriting any correct initialization provided by the Sketch. "Islands of Correctness" cannot survive the "Wavefront of Error."

---

## Final Recommendation: The Bidirectional Pivot

The "Fractal Computer" is viable, but it requires a specific architecture:
1.  **Bidirectional Attention (BERT/Diffusion):** To allow Token $i$ to see "Future" Token $j$ (if $j$ is a Sketch anchor). This enables the "Islands" to merge.
2.  **Masked Training:** The model must be trained to predict $Token_i$ given a noisy context, not just a clean prefix.

**The "Fractal" approach—combining Sketching (Structure) with Flash Flood (Speed)—remains the correct path for high-performance AI, but it must be built on a non-causal foundation.**
