# Research Log: Fractal Falsifiable Diffusion Engine

Testing Chen's 2025 paper on discrete diffusion with energy-based hallucination detection.

---

## Phase 1: Synthetic Test (2024-11-25)

**Task**: root → [left, right] tree expansion (depth 3, 8 leaves)
**Model**: 1.2M params, discrete diffusion on Boolean hypercube
**Result**: **100% accuracy** - model learned deterministic mapping perfectly
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

## Summary

| Phase | Task | UNK Rate | Detection | Verdict |
|-------|------|----------|-----------|---------|
| 1 | Synthetic tree | 0% | N/A | PASS (training) |
| 2 | Shakespeare next-char | 78% | 52.5% | FAIL |
| 2.5 | BPE decompress | 0% | 60/32/51% | FAIL |

**Hypothesis**: Chen's Lemma 7 energy bound may not transfer to discrete diffusion, or requires underfit models / continuous data.
