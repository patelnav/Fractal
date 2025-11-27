# Research Log: Fractal Falsifiable Diffusion Engine

Testing Chen's 2025 paper on discrete diffusion with energy-based hallucination detection.

---

## Phase 1: Synthetic Test (2024-11-25)

**Task**: 3-level synthetic hierarchy, root → chunks → fine tokens (deterministic expansions)
**Model**: ~1.2M params, discrete diffusion on Boolean hypercube
**Result**: **100% accuracy** - model learned all deterministic mappings perfectly
**Hallucination test**: Not run (deferred to real data)

---

## Phase 2: Shakespeare Next-Char (2024-11-25)

**Task**: Given root token, predict next 16 characters
**Data**: TinyShakespeare, word-level tokenizer (1000 vocab)
**Model**: 5.4M params, trained to loss ~2.8
**Hallucination test**: **FAILED - 52.5% detection** (random chance)

Root cause: 78% UNK tokens in roots made energy metric meaningless

---

## Phase 2.5: BPE Decompression (2024-11-26)

**Task**: BPE token → character sequence (deterministic ground truth)
**Data**: TinyShakespeare, BPE vocab 512, 503K samples, **0% UNK**
**Model**: 5.4M params, trained to loss 0.0000 (perfect)

**Hallucination tests**:
| Test | Detection | Status |
|------|-----------|--------|
| Energy (random wrong) | 60% | PARTIAL |
| Diversity (gen variance) | 32% | FAIL |
| Cross-token (mismatch) | 51% | FAIL |

**Conclusion**: Energy-based detection fails even with 0% UNK and perfect training. Cross-token at 51% shows model doesn't learn token-specific energy wells.

---

## Phase 3: Contrastive BPE Energy Head

**Task**: Train an explicit energy head to distinguish valid vs invalid BPE→chars pairs on the Phase 2.5 decompression task.
**Data**: Same TinyShakespeare BPE decompression dataset (512 vocab, 503K samples, **0% UNK**)
**Model**: Phase 2.5 transformer + 2-layer MLP energy head; joint diffusion + contrastive energy loss
**Hallucination test**: **SUCCESS – 100% detection** on held-out mismatched pairs at a fixed 0.5 energy threshold

Key idea: Replace the implicit Chen energy integral with an amortized energy head trained on contrastive (correct, wrong) pairs, yielding clear separation (E≈0 for correct, E≈1 for wrong).

---

## Phase 4: Integrated Fractal Engine

**Task**: Shared-weight hierarchical diffusion over two abstraction levels (roots → chunks, chunks → chars) with energy-based hallucination detection at each level.
**Data**: Hierarchical BPE over Shakespeare: Level 2 chars (65), Level 1 chunks (1024), Level 0 roots (2048); ~349K L0 samples, ~421K L1 samples; effectively **0% UNK**
**Model**: ~6M param transformer with unified vocabulary, 2-value level embedding, separate heads per level, shared contrastive energy head
**Hallucination tests**:
| Level | Detection | Status |
|-------|-----------|--------|
| Root → Chunks (L0) | 100% | SUCCESS |
| Chunk → Chars (L1) | 99% | SUCCESS |

Result: Confirms the fractal/shared-weight hypothesis and shows a single energy head can verify expansions at multiple levels.

---

## Phase 5: Dreamer Demo (Vertical Generation)

**Task**: Use the Phase 4 engine + energy head for generative decoding via rejection sampling (root → chunks → chars) instead of pure decompression.
**Setup**: Given a root ID, repeatedly sample noisy expansions and accept only low-energy (<0.5) candidates at both levels.
**Result**: **Perfect reconstruction** on tested roots and chunks; rejection sampling yields correct text for in-distribution seeds, demonstrating self-verifying top-down ("vertical") generation.

---

## Phase 6: Hybrid Fractal Engine (Manager + Fractal)

**Task**: Combine a small autoregressive “Manager” GPT that samples root sequences with the fractal engine that renders each root into text, with energy-based verification.
**Data**: 350K root-token sequences from the Phase 4 hierarchy (Shakespeare).
**Model**: ~1M-param 4-layer GPT manager + ~6M-param fractal engine with contrastive energy head; generation pipeline: Manager roots → fractal rejection-sampled expansions.
**Result**: **Novel Shakespeare-like passages** with active rejection (multiple retries per root), qualitatively coherent and grammatical; demonstrates separation of plotting (AR) and rendering (diffusion + energy).

---

## Phase 7-10: Ouroboros System (Reasoning Verification)

**Objective:** Apply the "Contrastive Energy Head" concept to verify Chain-of-Thought reasoning on GSM8K.

**Method:**
1.  **Phase 7:** Train `OuroborosModel` (BERT-like verifier) on Generator outputs (Gemma-2B). Loss = Contrastive Energy (push down correct, push up incorrect).
2.  **Phase 8:** Use "System 2" inference (generate 16 candidates -> select lowest energy).
3.  **Phase 9:** "Adversarial Hardening" - mine hard negatives from the generator to toughen the verifier.
4.  **Phase 10:** Full system evaluation on GSM8K Test Set.

**Results (Phase 10):**
*   **Baseline (Gemma-1B):** 52.46% (Pass@1)
*   **Oracle (Pass@16):** 80.14% (Potential Ceiling)
*   **Ouroboros (Verifier):** 50.42% (Pass@1)

**Conclusion:**
The verifier failed to outperform the baseline (-2.04%).
**Failure Mode:** The verifier learned to assign extremely low energy (high confidence) to "degenerate" artifacts like repetition loops or plausible-looking but mathematically wrong answers. The "hardening" phase was insufficient to close these holes in the energy landscape.
**Lesson:** Pure energy minimization without structural constraints or rule-based sanity checks is fragile. Future work should incorporate Process Reward Models (PRMs) or explicit rule-based filtering.

---

## Summary

| Phase | Task | UNK Rate | Detection | Verdict |
|-------|------|----------|-----------|---------|
| 1 | Synthetic tree | 0% | N/A | PASS (training) |
| 2 | Shakespeare next-char | 78% | 52.5% | FAIL |
| 2.5 | BPE decompress | 0% | 60/32/51% | FAIL |
| 3 | Contrastive BPE energy | 0% | 100% | PASS (energy head) |
| 4 | Fractal engine L0/L1 | 0% | 100% / 99% | PASS (multi-level) |
| 5 | Dreamer demo (gen) | 0% | N/A | PASS (vertical gen) |
| 6 | Hybrid manager+fractal | 0% | N/A | PASS (hybrid gen) |
| 7-10 | GSM8K Reasoning | N/A | 50% (vs 52% base) | FAIL (verifier fragility) |

**Final Verdict:** The "Fractal/Energy" hypothesis works brilliantly for **deterministic decompression** (Phases 3-4) where ground truth is absolute. It struggles significantly in **open-ended reasoning** (Phases 7-10) where the "correctness" manifold is complex and the verifier can be tricked by plausible-sounding hallucinations or artifacts.

---

## Strategic Pivot (2025-11-26): From "Soft" to "Hard" Verification

**Decision:** Abandon Vector 3 (Text/Math Reasoning) and pivot to **Vector 6 (Program Synthesis & Execution)**.

**Rationale:**
1.  **The Failure of Soft Verification:** In Phase 10, the verifier assigned low energy (high confidence) to confident hallucinations and repetition loops. In the domain of natural language reasoning, "plausible" is too close to "correct" in the embedding space.
2.  **The Promise of Hard Verification:** We need a domain where correctness is binary and irrefutable. Code execution provides this. A function either passes its tests or it doesn't.
3.  **Infinite Ground Truth:** Unlike GSM8K (finite dataset), we can generate infinite synthetic coding problems and run the solutions to get perfect `(Prompt, Code, Pass/Fail)` triplets without human labeling.

**New Objective (Phase 11+):** Train an energy-based verifier to predict **Execution Validity**.
*   **Input:** Problem Spec + Generated Code
*   **Training Target:** Energy ≈ 0 if Tests Pass, Energy ≈ 1 if Tests Fail.
*   **Goal:** Use the verifier to reject invalid code *before* execution (or to guide search), solving the "Plausible but Wrong" problem by grounding it in compiler reality.