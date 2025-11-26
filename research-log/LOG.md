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

**Updated hypothesis**: Chen-style implicit energy (∫||score||² dt) fails as a practical hallucination detector for discrete diffusion on lookup-style tasks, but explicit contrastive energy heads + hierarchical fractal structure can achieve near-perfect detection and enable self-verifying generation.
