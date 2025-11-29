# JSON Repair Engine (Product Concept)

## Current Status (2024-11)

| Stage | Description | Status |
|-------|-------------|--------|
| 0 | Baseline & Evaluation Harness | **Done** |
| 1 | JSON Denoiser Training | **Done** (REINFORCE) |
| 2 | Energy Head Training | Skipped (REINFORCE worked better) |
| 3 | Inference Repair Loop | **Done** (98.5% parse success) |
| 4 | Productization (Library + CLI) | In Progress |
| 5 | Real-world Benchmarking | Not Started |

### Benchmark Results (Stage 3)

| Method | Parse@1 | Locality | Avg Edits |
|--------|---------|----------|-----------|
| Do Nothing | 37.2% | 64.1% | 0.0 |
| Heuristic | 88.3% | 76.3% | 11.6 |
| **Denoiser (full)** | **98.5%** | **98.9%** | **0.4** |

**Key finding**: Full-sequence denoising outperforms window-based iterative repair. The REINFORCE-trained 8-layer model (26M params) achieves near-perfect results with minimal edits.

## Overview

The JSON Repair Engine is a **high-accuracy, minimal-edit JSON repair tool** built on top of the Phase 31 Universal Denoiser + Energy Head. It takes large, broken JSON blobs (typically from LLMs or messy configs), localizes structural errors, and proposes **minimal, structurally valid repairs** while preserving the rest of the file byte‑for‑byte.

Core properties:
- **Parser-backed correctness** – every candidate repair must pass a standard JSON parser.
- **Local edits only** – change as little as possible around the error; treat the rest as immutable.
- **Model-guided fixes** – a bidirectional denoiser and energy head learn realistic repair patterns from data, instead of relying only on hand-written heuristics.

Primary target users:
- Teams consuming JSON from LLMs (tools APIs, function calling, agents).
- Data/platform engineers dealing with large, occasionally-broken JSON configs/logs.
- Developers wanting a safer drop-in replacement for `json.loads` / `json.load`.

## Problem

Today, broken JSON is common and painful:
- LLM outputs are often “almost JSON” with missing commas, quotes, or truncated objects.
- Large config files (Kubernetes manifests, app configs) are edited by hand and easily corrupted.
- Existing tools either:
  - Apply **hand-written heuristics** that break on more complex or stacked errors, or
  - Ask a **general LLM to rewrite the entire blob**, losing locality and making diffs hard to trust.

We want a tool that:
- **Repairs** JSON instead of rewriting it.
- **Guarantees validity** via a parser-in-the-loop.
- **Explains what changed**, with small, inspectable diffs.

## Solution Overview

At a high level, the JSON Repair Engine does:
1. **Error localization**
   - Run a standard JSON parser (`json`, `orjson`, etc.).
   - On failure, capture the error location (line, column, char offset).
   - Define a **local window** around that position (±N tokens/chars).

2. **Tokenization & masking**
   - Tokenize into JSON tokens: `{`, `}`, `[`, `]`, `:`, `,`, strings, numbers, booleans, `null`.
   - Mark tokens inside the window as **editable**, outside as **anchors**.
   - Replace editable tokens with `<MASK>` before sending to the denoiser.

3. **Refinement (Universal Denoiser)**
   - Use a JSON-trained Universal Denoiser in edit mode:
     - Input: full token sequence with a masked window.
     - At each refinement step:
       - Predict all positions.
       - Re-anchor non-masked tokens so they never drift.
   - After K steps (e.g., 2–5), decode back to JSON text.

4. **Validation & iteration**
   - Parse the repaired JSON:
     - If parse succeeds: accept (optionally run schema checks).
     - If parse fails: take the new error location, define a fresh window, and repeat (up to K attempts).
   - Optionally generate a small **beam** of candidate repairs for a given window and:
     - Filter by parse success.
     - Rank by energy score and minimal edit distance.

5. **Output & integration**
   - Return:
     - The repaired JSON.
     - A diff (original vs repaired) for inspection.
   - Provide both a **Python library** API and a **CLI** (`json-repair broken.json > fixed.json`).

## Differentiation

Compared to existing heuristic libraries (e.g. `json_repair` and similar ports in other languages), this engine aims to:
- Learn repairs from large synthetic corpora instead of relying solely on hand-written rules.
- Handle **multi-error** and more subtle structural issues via iterative localization.
- Provide stronger **locality guarantees** (≥99.5% of tokens outside a small window unchanged).
- Offer a learned **energy head** that distinguishes valid vs broken sequences and helps rank candidates.

The parser still acts as a hard validator; the model proposes, the parser decides.

## Technical Backbone

The JSON Repair Engine is an application of the **Universal Refiner + Critic** pattern:
- **Universal Denoiser** (from Phase 31):
  - A bidirectional transformer trained on `(corrupted_tokens, clean_tokens, sigma)` triples.
  - Specialized here for JSON: JSON-specific vocabulary and corruption regimes biased toward local edits.
- **Energy Head**:
  - A small head scoring sequences for “validity”.
  - Trained on:
    - Positives: clean JSON and successful repairs (parse OK, small edit distance).
    - Negatives: synthetically corrupted JSON and failed repair attempts.
  - Used to rank candidate repairs and optionally to reject obviously bad outputs even if they parse.

The core hypothesis:
- With realistic corruption training and a parser in the loop, the denoiser+energy system can achieve:
  - **ParseSuccess@1 ≥ 90%** on realistic corruptions.
  - **ParseSuccess@K ≥ 95%** for a small number of localized iterations (K ≤ 3).
  - Extremely high locality (≥99.5% of tokens outside a radius W unchanged).

## Product Surfaces

Target surfaces:
- **Python library**
  - `repair_json_str(bad_json: str) -> str`
  - `repair_json_file(path: str) -> str`
  - Options for:
    - Returning a Python object vs JSON string.
    - Strict vs permissive modes.
    - Max iterations / beams / timeouts.

- **CLI**
  - `json-repair INPUT.json > OUTPUT.json`
  - Flags:
    - `--inline`: repair file in-place.
    - `--diff`: print unified diff instead of the full JSON.
    - `--max-iters`, `--beam-size`, `--strict`.

- **Editor / IDE integration (later)**
  - VS Code extension or LSP command:
    - “Fix JSON in this buffer” → applies a minimal patch.
  - Good for debugging large configs and LLM responses in dev environments.

- **LLM integration**
  - A drop-in guardrail for tools/function-calling stacks:
    - Capture raw model output.
    - Run through JSON Repair Engine.
    - Only pass structurally valid JSON downstream.

## Build Stages / Roadmap

This roadmap assumes the Phase 31 Universal Denoiser infrastructure already exists.

### Stage 0 – Baseline & Evaluation Harness

Goal: Establish strong baselines and measurement before training the model.

- Build a **JSON corruption engine**:
  - Start from real JSON corpora (configs, manifests, logs).
  - Inject single and multi-error corruptions (missing commas/braces, truncated strings, extra garbage, etc.).
- Implement **evaluation harness**:
  - Metrics:
    - `ParseSuccess@1`, `ParseSuccess@K`.
    - `EditSize` and locality (fraction of unchanged tokens outside a window W).
    - Wall-clock time per repair.
- Implement **baselines**:
  - Simple heuristic fixer (insert/remove common punctuation around parser error).
  - Existing heuristic library baseline (e.g., `json_repair`).
  - Strong LLM “rewrite this JSON” baseline.

Deliverable: `benchmark.py` with baseline numbers and a clear gap to target.

### Stage 1 – JSON Denoiser Training

Goal: Train a JSON-specific Universal Denoiser that can repair local corruptions.

- Implement `data_json.py`:
  - Corpus loader, tokenizer hooks, corruption pipelines.
- Implement `tokenizer_json.py`:
  - Minimal JSON-aware tokenizer (tokens for braces, brackets, colon, comma, literals).
- Implement `model_denoiser.py`:
  - Phase 31 denoiser architecture adapted to JSON vocab.
  - Training objective: denoising on local corruptions (avoid full-mask generation).
- Implement `train_denoiser.py`:
  - Training loop for `(corrupted_tokens, clean_tokens, sigma)`.
  - Validation set with corruption patterns that match Stage 0.

Deliverable: Checkpoints with strong token-level reconstruction accuracy and encouraging repair behavior in small-scale tests.

### Stage 2 – Energy Head Training

Goal: Add a critic that scores candidates and helps reject or rank repairs.

- Implement `model_energy.py`:
  - Small head on top of the denoiser’s sequence representation.
- Implement `train_energy.py`:
  - Construct:
    - Positives: clean JSON + successful denoiser repairs.
    - Negatives: corrupted sequences + failed repair attempts + adversarial invalid JSON.
  - Train with contrastive / BCE loss.
- Add evaluation metric:
  - ROC-AUC on valid vs invalid JSON sequences (target ≥ 0.95).

Deliverable: An energy model that can reliably separate valid-looking JSON from broken or suspicious outputs.

### Stage 3 – Inference Repair Loop ✅ COMPLETE

Goal: Build the full parser → window → edit → parse loop for real JSON files.

**Implemented** in `inference_repair.py`:
- `repair_json_full_denoise()` - One-shot full-sequence repair (recommended, 98.5% success)
- `repair_json()` - Iterative window-based repair with:
  - Hard locality enforcement (anchored tokens)
  - Progressive window expansion
  - Edit region tracking
- `repair_json_beam()` - Beam search with confidence-based ranking

**Key learning**: Full-sequence denoising works much better than window-based masking for this model. The REINFORCE-trained model expects to see the full corrupted sequence and denoise all positions simultaneously, not fill in masked windows.

**Results**:
- 98.5% parse success (vs 88.3% heuristic baseline)
- 98.9% locality (only 0.4 tokens changed on average)
- 23ms average repair time

Deliverable: ✅ `repair_json_full_denoise()` achieves target metrics.

### Stage 4 – Productization (Library + CLI)

Goal: Turn the engine into a practical tool for users and systems.

- Library API:
  - Ergonomic wrapper functions:
    - `repair_json`, `loads`, `load`, `from_file`-style interfaces.
  - Configurable:
    - Strict vs permissive.
    - Max iterations, timeouts, beam size.
    - Optionally return diff and/or repair logs.
- CLI:
  - `json_repair_cli.py`:
    - Supports stdin/stdout, inline file edits, diff mode, and basic knobs (iters, strict).
  - Packaging for `pipx` / `pip`.

Deliverable: A usable package that can be dropped into existing Python codebases and tooling pipelines.

### Stage 5 – Benchmarking, Tuning, and Launch

Goal: Validate SOTA performance and refine UX.

- Run extensive benchmarks:
  - Synthetic corruption suite from Stage 0.
  - Real-world JSON from LLM traces and production logs (where possible).
  - Compare against:
    - Heuristic fixer(s).
    - Existing `json_repair`-style library baseline.
    - LLM-based JSON repair.
- Tune:
  - Window size, number of refinement steps, beam size.
  - Energy thresholds and ranking strategies.
- Document results in `RESULTS.md`:
  - Final metrics:
    - `ParseSuccess@1`, `ParseSuccess@K`.
    - Locality and edit sizes.
    - Speed vs baselines.
  - Qualitative examples and failure cases.

Deliverable: A set of clear, reproducible metrics demonstrating that the JSON Repair Engine offers a strong practical improvement over existing tools on real JSON repair workloads.

## Long-Term Extensions

Once JSON repair is solid, the same pattern extends naturally to:
- **JSON5 / JSON with comments** and other dialects.
- **YAML, TOML, and config languages** with well-defined grammars.
- **Hierarchical editing** for more complex structured documents (e.g., AST-level code repair).

JSON is the proving ground: if the Universal Refiner + Critic works well here, it is a strong validation of the broader “fractal editor” story.

