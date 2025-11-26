# Project: The Fractal Falsifiable Diffusion Engine

**Objective:** Validate a hyper-sample-efficient, recursive architecture for discrete generative AI.
**Core Philosophy:** "Structure is the Signal." Standard LLMs treat data as flat strings; we treat it as hierarchical **Trees** (Roots → Chunks → Fine Details).

---

## 1. The Theoretical Breakthrough
Based on Yuansi Chen’s *Perturbed Reverse Heat Process* (2025).
*   **The Problem:** Standard diffusion models rely on Gaussian noise, which doesn't map natively to discrete data (text/code).
*   **The Solution:** Chen proves that the Heat Semigroup on the Boolean Hypercube allows for exact, mathematically bounded discrete diffusion.
*   **The Pivot:** Early experiments (Phase 2) showed that Chen's "Implicit Energy" was too subtle for robust hallucination detection. We successfully pivoted to an **Explicit Contrastive Energy Head** (Phase 3), which learns to distinguish valid vs. invalid structural expansions with near 100% accuracy.

## 2. The Hybrid Fractal Architecture
We have validated a novel 3-part architecture (proven in Phase 6 "Hybrid"):

1.  **The Manager (Autoregressive "Plotter"):**
    *   A tiny GPT that generates the high-level "Skeleton" (Roots).
    *   *Role:* Global coherence and trajectory planning.

2.  **The Fractal Engine (Discrete Diffusion "Renderer"):**
    *   A shared-weight model that recursively expands Roots → Chunks → Fine Tokens.
    *   *Key Feature:* **Holographic Learning**. The model uses the same weights to expand a "Chapter" as it does a "Sentence," learning structural isomorphisms.
    *   *Key Feature:* **Flash Flood Decoding**. Because rendering is decoupled from plotting, we can expand massive trees in parallel (SOTA Speed potential).

3.  **The Energy Head (Verifier "Critic"):**
    *   A discriminator trained on contrastive pairs.
    *   *Role:* **Ouroboros Reasoning**. It scores every expansion step. If energy is high (invalid logic/structure), the model backtracks and retries. This enables "System 2" thinking and zero-hallucination generation.

## 3. Research Status & Learnings
*   **Phase 1 (Synthetic):** $\checkmark$ Confirmed model can learn strict hierarchical mappings.
*   **Phase 4 (Fractal Structure):** $\checkmark$ Confirmed Universal Refinement Hypothesis—one model can handle multiple levels of abstraction.
*   **Phase 6 (Hybrid System):** $\checkmark$ Confirmed the Manager+Fractal split works, producing coherent outputs with active self-verification.

## 4. Future Directions
We are now applying this architecture to specific domains including Coding, World Models, and Governance.

**For the detailed list of exploration vectors and the scoring rubric, see [Vectors/README.md](Vectors/README.md).**
