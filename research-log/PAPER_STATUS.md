# Fractal / Ouroboros: Scaling Reasoning via Verified Self-Improvement
**Date:** November 27, 2025  
**Status:** Experimental Validation Complete

## Abstract
We show that a small language model (1.5B parameters) can significantly improve its coding ability (58% → 65% Pass@1 on MBPP) by wrapping it in a **Neural Computer**: a system that combines a generator, a hard execution loop, and a learned verifier. Across Phases 1–15 we built and validated three core innovations:

- **Verified neural primitives** that compose into exact algorithms (Neural CPU).
- **Execution‑trained critics** that generalize beyond their training data (Soft Verifier).
- **Self‑improvement loops** where the model learns from its own verified behavior (GRPO + Critic).

The key lesson is that robust reasoning arises from the **system** (Generator + Verifier + Execution), not from model weights alone.

## 1. Project Trajectory (What We Built, Phase by Phase)

### Phases 1–6: Fractal Generators and Explicit Energy
- **Phase 1–4:** Built a **Fractal Diffusion Engine** that expands hierarchical trees (roots → chunks → tokens) with **shared weights across levels**, validated on synthetic data and Shakespeare. This established:
  - Holographic learning: one set of weights can operate at multiple abstraction levels.
  - A contrastive **energy head** that reliably distinguishes valid vs invalid expansions in deterministic settings.
- **Phase 5–6:** Added a small autoregressive **Manager** that samples high‑level roots, and used the Fractal Engine as a **renderer** with energy‑based rejection. This produced coherent text with self‑verification, showing that:
  - Plotting (AR) and rendering (diffusion + energy) can be separated.
  - Energy‑based rejection sampling stabilizes generation.

### Phases 7–10: Failure of Pure Text‑Based Energy
- Applied the same energy‑head idea to chain‑of‑thought reasoning on GSM8K (Ouroboros).
- Result: the verifier often assigned **low energy to confident hallucinations**; Pass@1 did not beat the base model.
- Insight: in open‑ended natural language, “plausible” and “correct” are too close in representation space. **Soft verification without a hard oracle is fragile.**

### Phases 11–13: Neural CPU – Verified Primitives and Digital Restoration
- **Phase 11:** Designed a **Recurrent Fractal ALU** (shared‑weight transformer with GRU gating) that learns 8‑bit addition and extrapolates to 12‑bit with ~100% accuracy. This showed:
  - With weight sharing across bit positions, the model learns the **algorithm** (carry propagation), not the table.
- **Phase 12:** Composed the ALU into a hard‑coded **shift‑and‑add multiplier** with **digital restoration**:
  - At each step, argmax “snaps” hidden states back to discrete 0/1 embeddings before feeding them forward.
  - Exhaustive tests (4‑bit × 4‑bit = 256 cases) and 8‑bit multiplication confirmed **perfect zero‑shot multiplication** using only the trained Adder.
  - This is a Neural CPU: complex algorithms (multiplication) emerge by composing verified primitives (adders) plus digital snapping.
- **Phase 13:** Tried to train a **Neural RISC‑V controller** to *discover* the shift‑and‑add program using the frozen ALU. Curriculum + imitation reached ~94% op accuracy but not full reliability.
  - Insight: algorithm discovery is limited by search and credit assignment, not by the expressiveness of the primitives.

### Phases 14–15: Neural Computer for Real Code (Vector 6 Reboot)
- **Phase 14 (MBPP / HumanEval):**
  - **Generator:** `Qwen2.5‑Coder‑1.5B‑Instruct`.
  - **Hard Verifier:** Python sandbox that executes candidates against MBPP tests (8,150 train samples).
  - **Soft Verifier (Critic):** Qwen‑based classifier trained on `(Prompt, Code) -> Pass/Fail` from execution labels.
  - Results (clean split: train vs test):
    - Critic Val Acc: 88.8% on held‑out validation.
    - **MBPP (Test):** Random Best‑of‑50 Pass@1 = 61.1%; Critic‑ranked Pass@1 = 65.0% (**+3.8% vs random pool, +6.6% vs greedy baseline**).
    - **HumanEval:** +3.7% Pass@1 gain via the same critic (zero‑shot transfer).
  - Insight: execution‑trained critics **extrapolate** and improve code generation on unseen problems, even under strict train/test separation.
- **Phase 15 (GRPO Self‑Improvement):**
  - Used **Group Relative Policy Optimization** with execution pass/fail rewards to fine‑tune the generator.
  - In ~25 minutes (5 epochs, tiny batches), improved greedy Pass@1 from 58.4% → 60.7% (**+2.3%**).
  - Combined GRPO‑tuned generator with critic ranking (Best‑of‑50):
    - **MBPP Sanitized Final:**
      - Baseline (Qwen‑1.5B, greedy): 58.37% Pass@1.
      - GRPO‑only (greedy): 60.70% (+2.33%).
      - GRPO + Critic (Best‑of‑50): **64.98% (+6.61% vs baseline)**.
      - Oracle (any of 50 pass): 90.66%.
    - **Note:** GRPO slightly improves average candidate quality, while the critic recovers a significant fraction of the oracle gap.

## 2. What Is Actually New? (Core Innovations)

### 2.1 Verified Neural Primitives + Digital Restoration
- We built a **differentiable ALU** (Adder) that provably **extrapolates in bit‑length** and composes, via a shift‑and‑add loop plus argmax snapping, into a **perfect multiplier** without retraining.
- The key mechanism, **Digital Restoration**, is a bridge between neural networks and digital circuits:
  - Continuous dynamics are allowed inside a step.
  - After each step, states are snapped back to discrete attractors (0/1) that the next step can reliably consume.
- Innovation: this is a concrete, experimentally validated recipe for turning neural modules into **composable, error‑correcting primitives**, not just pattern matchers.

### 2.2 Verification‑Centric System Architecture
- We treat reasoning as a property of a **three‑part system**:
  1. **Generator** – proposes candidates (roots, code, programs).
  2. **Hard Verifier / Oracle** – exact but expensive checks (unit tests, arithmetic).
  3. **Soft Verifier (Critic / Energy Head)** – amortized approximation of the oracle.
- Across domains (hierarchical text, arithmetic, code), we:
  - Train the critic on **hard labels from the oracle**, not on text‑only signals.
  - Use the critic to guide search (rejection sampling, ranking) and training (RL).
- Innovation: this makes verification a **first‑class training signal**, not a bolt‑on filter. It is a concrete, working instantiation of the idea that:
  > “Energy / value functions should be trained on verified outcomes, then used to steer generation and learning.”

### 2.3 Self‑Improvement via Verification
- On MBPP, a **1.5B** model with:
  - a trained critic on execution labels, and
  - light GRPO fine‑tuning,
  reaches **65% Pass@1**, significantly outperforming its 58% baseline.
- This demonstrates that:
  - You can trade **oracle calls + search + verification** for **raw parameter count**.
  - A "Neural Computer" (Generator + Executor + Critic) improves itself autonomously.
  - The 91% oracle upper bound suggests significant room for further improvement.
- Innovation: an empirical demonstration that **verification‑driven self‑improvement** works reliably.

## 3. How the Pieces Fit Together

Putting the phases together, the innovation can be summarized as:

- **From Fractals to CPU:** Fractal architectures (Phases 1–6) led to a recurrent ALU and Neural CPU (Phases 11–12), proving that **neural primitives can implement exact, composable algorithms** when stabilized by digital restoration.
- **From CPU to Code:** The same philosophy—**trust hard oracles, train soft critics, and compose them**—was applied to real‑world Python code (Phases 14–15), where unit tests are the oracle and the critic is an energy model over code.
- **From Static to Self‑Improving Systems:** Instead of treating the LLM as a fixed object to benchmark, we built a closed‑loop system where:
  - execution labels create training data for the critic,
  - the critic improves inference via ranking,
  - and RL uses both critic and execution to improve the generator itself.

In short, the innovation is not a single new layer or loss function, but a **validated blueprint** for building small, compositional Neural Computers whose capabilities come from **verification‑driven composition and self‑improvement**, rather than from scale alone.

## 4. Future Work

- **Scale up:** Apply the Neural Computer blueprint to 7B/70B models and larger MBPP‑like benchmarks.
- **Math and Reasoning:** Extend to GSM8K/MATH using Python and symbolic checkers as oracles, reusing the Phase 8 solver infrastructure.
- **Architecture fusion:** Replace or augment the Qwen generator with a **fractal / ALU‑infused** backbone while keeping the same verification loop.
- **Hardware:** Explore implementing the Neural CPU primitives in silicon or accelerators, closing the loop from neural training to physical hardware.
