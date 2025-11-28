# Phase 32: JSON Planner-Refiner

## Objective

Apply the validated **AR Planner + Bidirectional Refiner + Energy Critic** architecture to JSON/YAML configuration generation.

This directly tests the Phase 31 conclusion: bidirectional models are excellent refiners but cannot generate structure from scratch. The AR planner provides the structural skeleton; the bidirectional refiner fills values and supports edits.

---

## Architecture

```
AR Planner          →  Bidirectional Refiner  →  Energy Critic
(generates skeleton)   (fills values)            (scores validity)
```

### 1. AR Planner (Skeleton Generator)
- **Input:** Natural language spec or empty prompt
- **Output:** JSON skeleton with `<VALUE>` placeholders
- **Vocab:** `{`, `}`, `[`, `]`, `:`, `,`, `"`, `<KEY>`, `<VALUE>`, whitespace, common keys
- **Architecture:** Small causal transformer (2-4 layers)
- **Example output:**
  ```json
  {"<KEY>": <VALUE>, "<KEY>": [<VALUE>, <VALUE>]}
  ```

### 2. Bidirectional Refiner (Value Filler)
- **Input:** Skeleton from planner
- **Output:** Complete JSON with actual values
- **Architecture:** Bidirectional transformer (reuse Phase 31 UniversalDenoiser)
- **Training:** Masked denoising on (skeleton, complete) pairs
- **Example:**
  ```json
  {"name": "config", "ports": [8080, 443]}
  ```

### 3. Energy Critic
- **Scoring dimensions:**
  - Parse validity (JSON.parse succeeds)
  - Schema conformance (if schema provided)
  - Simple constraint checks (types, ranges)
- **Architecture:** Same as Phase 31 energy head
- **Training:** Contrastive on (valid, invalid) JSON pairs

---

## Data

### Synthetic JSON Generator
Generate diverse JSON configs with:
- Nested objects (depth 1-4)
- Arrays (0-10 elements)
- Value types: strings, numbers, booleans, null
- Key patterns: camelCase, snake_case, common config keys

### Corruption Engine
- Swap values between keys
- Remove/add commas, brackets
- Type mutations (string→number, etc.)
- Truncation

### Dataset Size
- Training: 50K samples
- Validation: 5K samples
- Test: 2K samples

---

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Parse Valid | JSON.parse succeeds | >95% |
| Schema Valid | Matches expected schema | >90% |
| Exact Match | Skeleton→values matches ground truth | >70% |
| Edit Stability | Anchor preservation on local edits | 100% |
| Repair Accuracy | Fix corrupted JSON | >85% |

---

## Stages

### Stage 1: Data Pipeline (Local)
- [ ] Implement JSON generator with configurable complexity
- [ ] Implement skeleton extractor (JSON → skeleton with placeholders)
- [ ] Implement corruption engine
- [ ] Generate train/val/test splits
- [ ] Verify data quality

### Stage 2: AR Planner (Cloud GPU)
- [ ] Define skeleton vocabulary
- [ ] Train small causal model to generate skeletons
- [ ] Benchmark skeleton validity rate
- **Gate:** >90% valid skeletons or debug

### Stage 3: Bidirectional Refiner (Cloud GPU)
- [ ] Adapt Phase 31 UniversalDenoiser for JSON
- [ ] Train on (skeleton, complete) pairs
- [ ] Benchmark generation quality
- **Gate:** >85% parse valid or debug

### Stage 4: Energy Critic (Cloud GPU)
- [ ] Generate contrastive pairs (valid, invalid)
- [ ] Train energy head
- [ ] Benchmark discrimination (ROC-AUC)
- **Gate:** >90% ROC-AUC or debug

### Stage 5: Integration & Benchmark
- [ ] Wire up full pipeline: Planner → Refiner → Critic
- [ ] Benchmark end-to-end generation
- [ ] Benchmark editing (anchor stability)
- [ ] Benchmark repair
- [ ] Compare to baseline (pure AR, pure MaskGIT)

---

## Success Criteria

**Phase 32 succeeds if:**
1. End-to-end JSON generation achieves >90% parse valid (vs Phase 31's 57% on arithmetic)
2. Editing preserves anchors at 100%
3. Repair achieves >85% accuracy
4. The Planner→Refiner split demonstrably outperforms pure bidirectional

**Phase 32 fails if:**
- AR planner cannot generate valid skeletons (>90%)
- Even with valid skeletons, refiner cannot fill values (>85% parse valid)

---

## Files to Create

```
phase32-json-planner-refiner/
├── PLAN_PHASE32.md          # This file
├── RESULTS.md               # Findings
├── data_json.py             # JSON generator, skeleton extractor, corruption
├── model_planner.py         # AR skeleton planner
├── model_refiner.py         # Bidirectional refiner (adapt from Phase 31)
├── model_energy.py          # Energy critic for JSON
├── train_planner.py         # Train skeleton generator
├── train_refiner.py         # Train value filler
├── train_energy.py          # Train energy head
├── inference_pipeline.py    # Full Planner→Refiner→Critic pipeline
├── benchmark.py             # Evaluation suite
└── run_*.sh                 # Execution scripts
```

---

## Relationship to Prior Work

- **Phase 31:** Proved bidirectional denoiser is refiner, not generator → motivates AR planner
- **Phase 6:** Manager + Renderer split → same architecture, new domain
- **Phase 4:** Energy head for verification → reuse pattern
- **Vector 6:** Path toward code (JSON is stepping stone)

---

## Next After Phase 32

If successful:
- **Phase 33:** Same pattern on small Python functions with execution-based evaluation
- Reuse planner/refiner/critic architecture, just change domain and energy function
