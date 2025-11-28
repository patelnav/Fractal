# Phase 32: JSON Repair Engine

## Objective

Build a **practical, high-accuracy JSON repair tool** powered by the Phase 31 bidirectional denoiser + energy head.

Given a large, broken JSON file (missing comma, quote, brace, etc.), the system should automatically:
1. Localize the error,
2. Propose a **minimal, structurally valid repair**,
3. Preserve all untouched regions byte-for-byte.

This is the first “real-world” deployment of the **Universal Refiner + Critic** pattern.

---

## Core Hypothesis

> A bidirectional denoiser trained on (corrupted_json, clean_json) pairs, plus a small energy head, can repair real-world JSON errors with **>95% parse success** while changing **only a tiny local region** around the error.

More concretely:
- Denoiser handles the **local structural repair**.
- Energy head helps reject obviously bad repairs.
- A parser in the loop provides a hard correctness check.

If this fails even on JSON, the “refiner as structural repair oracle” story is in trouble; if it works, it’s a strong practical validation.

---

## Architecture & Flow

High-level loop for a single JSON file:

1. **Error localization (parser layer)**
   - Run a standard JSON parser (e.g., `json` / `orjson` / `jsonc`).
   - On failure, capture the error position (line, column, char offset).
   - Define a **window** around that position (e.g., ±N tokens or characters).

2. **Tokenization & masking**
   - Tokenize the entire file into JSON tokens: `{`, `}`, `[`, `]`, `:`, `,`, string literals, numbers, booleans, `null`.
   - Mark:
     - Tokens **inside** the error window as editable.
     - Tokens **outside** as anchors (must not change).
   - Replace editable tokens with `<MASK>` before feeding to the denoiser.

3. **Refinement (Phase 31-style editor)**
   - Use a JSON-trained Universal Denoiser in **edit mode**:
     - Input: full token sequence with masked window.
     - At each refinement step:
       - Predict all positions.
       - Re-anchor non-masked tokens (outside the window + special tokens).
   - After K steps (2–5), decode tokens back to JSON text.

4. **Validation & iteration**
   - Try parsing the repaired JSON:
     - If parse succeeds: **accept** (optionally also run schema checks).
     - If parse fails: use the new error location to define a fresh window and repeat (cap attempts).
   - Optional: generate a small **beam** of repairs for a given window and:
     - Filter by parse success.
     - Rank by energy score or minimal edit distance.

5. **Output**
   - Emit the repaired JSON plus a diff (original vs repaired) for inspection.

---

## Data & Training

### JSON Corpus

- Collect a diverse corpus of valid JSON:
  - Open-source config files (Kubernetes manifests, package.json, tsconfig, etc.).
  - Synthetic configs from a simple JSON generator (nested objects/arrays, mixed types).
- Normalize to a maximum size (e.g., 2–8 KB per sample) for training; real files can be larger at inference via sliding windows.

### Corruption Engine

For each clean JSON sample, generate one or more corrupted versions:

- **Structural errors:**
  - Delete or insert commas, colons, braces, brackets, quotes.
  - Swap adjacent tokens.
  - Truncate strings or arrays.
- **Non-structural but valid “noise”:**
  - Change a value’s type (string ↔ number ↔ boolean).
  - Randomly perturb numbers or booleans.

Produce triples `(clean_tokens, corrupted_tokens, sigma)` just like Phase 31, but:
- Bias σ toward **local corruption** (e.g., 5–20% tokens in a window).
- Avoid full-mask; we’re explicitly in the “repair/edit” regime, not generation from scratch.

### Denoiser & Energy Head

- **Denoiser**: copy Phase 31 UniversalDenoiser architecture, but:
  - JSON-specific vocab and tokenizer.
  - Training only on repair-style corruption (no full-mask generation objective).
- **Energy head**:
  - Positives: clean JSON + “successful repairs” (parse OK, small edit distance).
  - Negatives: synthetically corrupted JSON and failed repair attempts.
  - Train with contrastive/BCE loss to score “valid-looking” sequences low, broken ones high.

---

## Evaluation & Success Criteria

On a held-out test set of corrupted JSON files:

1. **Primary metrics**
   - `ParseSuccess@1`: fraction of files that parse after a single repair run.
   - `ParseSuccess@K`: with up to K localized repair iterations (e.g., K=3).
   - `Locality`: fraction of tokens outside a fixed-radius window around the original error that remain unchanged (target ≈100%).
   - `EditSize`: average number of tokens changed (should be small).

2. **Secondary metrics**
   - `Energy ROC-AUC`: valid vs corrupted (target >0.95).
   - `Time per repair`: wall-clock vs baseline heuristics (e.g., “just delete offending char” or “try common fix patterns”).

3. **Baselines**
   - Simple heuristic fixer:
     - Try inserting/removing a comma/brace at the error location.
     - Re-parse; accept first success.
   - “LLM rewrite” baseline:
     - Prompt a strong code model to “fix this JSON” and compare:
       - Parse success.
       - Amount of unrelated change (often large).

**Success for Phase 32 (JSON Repair Engine):**

- `ParseSuccess@1 ≥ 90%` on realistic corruptions, and `ParseSuccess@K ≥ 95%` for K≤3.
- ≥99.5% of tokens outside a ±W token window around the error are unchanged.
- Energy head ROC-AUC ≥ 0.95 on valid vs invalid JSON.
- Clear win vs heuristic + LLM baselines on locality and reliability.

---

## Deliverables

- `phase32-json-repair/`
  - `PLAN_PHASE32.md` (this document)
  - `data_json.py` (corpus loader + corruption)
  - `tokenizer_json.py` (JSON tokenizer)
  - `model_denoiser.py` (JSON UniversalDenoiser)
  - `model_energy.py` (energy head)
  - `train_denoiser.py` (Stage 1)
  - `train_energy.py` (Stage 2)
  - `inference_repair.py` (repair loop: parser → window → edit → parse)
  - `benchmark.py` (metrics + baselines)
  - `RESULTS.md` (final numbers + qualitative examples)
- Optional:
  - `json_repair_cli.py`: simple CLI: `json-repair broken.json > fixed.json`.
  - Example VS Code / editor integration note.

---

## Relationship to Existing Phases

- **Phase 31**: Provided the core **refiner + energy head** pattern and proved it works for arithmetic repair/editing.
- **Phase 22 (Fractal Repair) / 17 (Hierarchical Edit)**: JSON repair is a concrete, high-value instance of those ideas.
- **Vectors 1 & 7**: This is an early, practical manifestation of “Fractal Editor / Hierarchical Editing,” but in a simpler, single-level domain (JSON trees).
