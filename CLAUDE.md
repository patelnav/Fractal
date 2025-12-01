# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Fractal Falsifiable Diffusion Engine** - A Python/PyTorch research project implementing discrete diffusion models for hierarchical text generation, based on Yuansi Chen's 2025 Perturbed Reverse Heat Process.

**Core Philosophy:** "Structure is the Signal" - treating data as hierarchical trees (Roots → Chunks → Characters) rather than flat strings.

**Current Status (Phase 30):** Validated that Flash Flood parallel decoding requires bidirectional attention (like BERT/Diffusion), not causal LLMs.

## Build & Run Commands

```bash
# Setup (use uv, not pip)
uv venv
source .venv/bin/activate
uv pip install torch numpy tqdm transformers datasets

# Run any phase - ALWAYS tee output to file
cd research-log/phase4-fractal-engine
python run_fractal_engine.py 2>&1 | tee /tmp/phase4_output.txt
# Then: tail -100 /tmp/phase4_output.txt

# Quick validation (any phase)
python run_*.py --device=cpu --max_iters=500 2>&1 | tee /tmp/quick_val.txt

# Test energy detection
python test_*.py 2>&1 | tee /tmp/test_output.txt
```

## Architecture

### Core Hybrid System

```
Planner (AR)          →  Renderer (Diffusion)    →  Critic (Energy)
Generates sketch/roots   Parallel expansion         Scores validity
```

**Key Discoveries:**
- **Holographic Learning**: Shared weights across abstraction levels work (Phase 4, 11)
- **Flash Flood**: 94x speedup validated (Phase 26), but requires bidirectional attention
- **Hard Verification**: Execution-based training drives extrapolation (Phase 14: +6% on MBPP)
- **Causal Collapse**: Causal LLMs destroy "Islands of Correctness" during parallel decoding (Phase 28-29)

### Hierarchical Data Representation

```
Level 0 (Roots):  ~2048 BPE tokens (semantic concepts)
Level 1 (Chunks): ~1024 BPE tokens (units)
Level 2 (Fine):   ~65 characters
```

## Key Files

| File | Purpose |
|------|---------|
| `PLAN.md` | Project vision & theoretical foundation |
| `research-log/LOG.md` | Complete research log (Phases 1-30) |
| `research-log/phase30-bidirectional-fractal/` | Latest: Bidirectional Flash Flood prototype |
| `research-log/phase26-flash-flood-scale/` | 94x speedup benchmark |
| `research-log/phase14-*/` | Code verification with GRPO |

## Code Patterns

### Discrete Diffusion Loop
```python
noised = add_noise(target, t)          # Add noise at timestep t
pred = model(condition, noised, t)     # Predict denoised
loss = cross_entropy(pred, target) + energy_loss
```

### Energy-Based Verification
```python
energy_correct = energy_head(embed_pair(condition, target))  # → 0
energy_wrong = energy_head(embed_pair(condition, wrong))     # → 1
# Rejection sampling: retry while energy > threshold
```

## Key Research Results

| Phase | Result | Insight |
|-------|--------|---------|
| 4 | 100%/99% detection | Universal Refinement Hypothesis confirmed |
| 11 | 99.5% OOD accuracy | Recurrent Fractal ALU extrapolates arithmetic |
| 12 | 100% zero-shot mult | Neural compositionality via digital restoration |
| 14 | +6% MBPP | Hard verification drives soft critic generalization |
| 26 | 94x speedup | Flash Flood throughput validated |
| 30 | 80% stability | Bidirectional attention preserves structure |

## Development Notes

- Run from phase directory (imports expect relative paths)
- Use `--device=cpu` for debugging, `--device=cuda` or `--device=mps` for speed
- Phases are self-contained with `run_*.py`, `test_*.py`, and `README.md`
- Reproducibility: set `random.seed()` and `torch.manual_seed()`
- **ALWAYS tee script output to `/tmp/`** and tell me where to tail it
- **Use `uv` for package management**, keep `requirements.txt` updated

## Lambda GPU Instance Management

**CRITICAL RULES - DO NOT VIOLATE:**
1. **NEVER terminate a Lambda instance before completing ALL requested tasks**
2. **If a task fails or is unclear, ASK before terminating - DO NOT assume it's optional**
3. **When given a checklist of tasks, complete EVERY item before shutdown**
4. **If you cannot complete a task, explicitly confirm with user before terminating**
5. **Archive/download results BEFORE termination, not during**

**Common Mistake to Avoid:**
- User requests: "run benchmark, create archive, download results, run analysis, then shutdown"
- WRONG: Complete some tasks, skip one because tool not found, terminate anyway
- RIGHT: Complete all tasks OR ask user about missing tasks BEFORE terminating

Lambda instances cost $1.29+/hr. Terminating early wastes both time and money if work must be redone.

## Feasibility Calibration

- **Read `FEASIBILITY_CALIBRATION.md`** for calibrated possibility intuitions
- Claude's training makes things *feel* impossible that are actually routine (Zone A)
- Key question: "Is this blocked by iteration cost (collapses) or something irreducible (Zone C)?"
- If something feels "too ambitious," ask: Zone A or Zone C blocker?
- Strategy shift: Explore broadly, try multiple approaches—iteration is cheap