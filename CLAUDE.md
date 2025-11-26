# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Fractal Falsifiable Diffusion Engine** - A Python/PyTorch research project implementing discrete diffusion models for hierarchical text generation, based on Yuansi Chen's 2025 Perturbed Reverse Heat Process.

**Core Philosophy:** "Structure is the Signal" - treating data as hierarchical trees (Roots → Chunks → Characters) rather than flat strings.

## Build & Run Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install torch numpy tqdm

# Run Phase 4 (most polished, ~10 min)
cd research-log/phase4-fractal-engine
python run_fractal_engine.py

# Run Phase 6 (full hybrid demo)
cd research-log/phase6-hybrid
python train_manager.py           # Train manager
python generate_hybrid.py         # Generate with verification

# Quick validation (any phase)
python run_*.py --device=cpu --max_iters=500

# Test energy detection
python test_*.py
```

## Architecture

### Three-Part Hybrid System (Phase 6)

```
Manager GPT (1M params)     →  Fractal Engine (6.3M params)  →  Energy Head
Autoregressive "Plotter"       Discrete Diffusion "Renderer"     Contrastive "Critic"
Generates root sequence        Root → Chunks → Characters        Scores validity (0=correct, 1=wrong)
```

**Key Features:**
- **Holographic Learning**: Shared weights across all abstraction levels (Level 0: root→chunks, Level 1: chunk→chars)
- **Flash Flood Decoding**: Parallel tree expansion since rendering is decoupled from plotting
- **Ouroboros Reasoning**: Energy head enables backtracking and rejection sampling for zero-hallucination

### Hierarchical Data Representation

```
Level 0 (Roots):  ~2048 BPE tokens (semantic concepts: "The king", "Exeunt")
Level 1 (Chunks): ~1024 BPE tokens (units: "The", "ing", " bear")
Level 2 (Fine):   ~65 characters
```

## Project Structure

```
research-log/                    # Phase-by-phase experiments
├── phase1-synthetic-test/       # Synthetic 1-to-4 tree (100% accuracy)
├── phase2-shakespeare/          # FAILED (52.5% - too many UNK tokens)
├── phase2.5-bpe-decompression/  # PARTIAL (60%)
├── phase3-contrastive-bpe/      # SUCCESS (100% - explicit contrastive energy)
├── phase4-fractal-engine/       # SUCCESS (100%/99% - shared weights proven)
├── phase5-dreamer/              # SUCCESS (rejection sampling generation)
└── phase6-hybrid/               # SUCCESS (Manager + Fractal Engine)

Vectors/                         # 9 future exploration directions
├── README.md                    # Vector descriptions & scoring rubric
└── *_EVALUATION.md              # Evaluations by Claude/Gemini/GPT

nanoGPT/                         # Karpathy's nanoGPT (base transformer)
data/shakespeare.txt             # Training data (1.1 MB)
checkpoints/                     # Trained models (~76 MB each)
```

## Key Files

| File | Purpose |
|------|---------|
| `PLAN.md` | Project vision & theoretical foundation |
| `TEST1.md` | Universal Refinement Hypothesis spec |
| `research-log/LOG.md` | Complete research log with results |
| `phase4-fractal-engine/run_fractal_engine.py` | Best working model |
| `phase6-hybrid/generate_hybrid.py` | Full generation pipeline |

## Code Patterns

### Discrete Diffusion Loop
```python
noised = add_noise(target, t)          # Add noise at timestep t
pred = model(condition, noised, t)     # Predict denoised
loss = cross_entropy(pred, target) + energy_loss
```

### Shared-Weight Multi-Level
```python
level_emb = level_embedding(level_id)  # 0 or 1
# Same transformer, different output heads per level
output = head_level0(out) if level == 0 else head_level1(out)
```

### Energy-Based Verification
```python
energy_correct = energy_head(embed_pair(condition, target))  # → 0
energy_wrong = energy_head(embed_pair(condition, wrong))     # → 1
# Rejection sampling: retry while energy > threshold
```

## Research Results

| Phase | Detection Rate | Notes |
|-------|---------------|-------|
| 3 | **100%** | Contrastive energy beats implicit Chen energy |
| 4 | **100%/99%** | Proved Universal Refinement Hypothesis |
| 6 | N/A | Novel text with active verification |

## Development Notes

- Run from phase directory (imports expect relative paths)
- Use `--device=cpu` for debugging, `--device=cuda` or `--device=mps` for speed
- Phases are self-contained with `run_*.py`, `test_*.py`, and `README.md`
- Reproducibility: set `random.seed()` and `torch.manual_seed()`
- can we add a clause whenever we are running any code or script or process, ALWAYS TEE IT TO A FILE. I do not want you tail/head/grep output without dumping it to a file first. because nothing is worse than re-running a long process because you didn't log the output