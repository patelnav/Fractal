# Phase 31: Universal Denoising Engine

## Core Thesis

A single **recurrent bidirectional diffusion model** trained on a **continuous noise curriculum** can unify:
- **Generation** (full mask → complete sequence)
- **Repair** (corrupted draft → fixed sequence)
- **Editing** (local mask → local fill)

All three are points on the same denoising curve. The model trades compute for quality at inference time via iterative refinement.

---

## Hypothesis to Falsify

> A recurrent bidirectional transformer, trained on (corrupted, clean) pairs at varying noise levels, achieves:
> 1. **>85% structural accuracy** on generation (vs Phase 30's 80.6%)
> 2. **>90% repair accuracy** when given 20% corrupted inputs
> 3. **100% anchor stability** when editing local regions (masked positions change, unmasked stay fixed)
> 4. **Single model** handles all three tasks without task-specific heads

If any of 1-4 fails, we learn which component is the bottleneck.

---

## Architecture

### Base: Recurrent Bidirectional Transformer

```
┌─────────────────────────────────────────────┐
│  Input: x_noisy (partially masked/corrupted)│
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  Token Embed + Position Embed + Noise Embed │  ← NEW: timestep/noise-level embedding
└─────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │   RECURRENT BLOCK     │ ← Same weights applied K times
        │  (Bidirectional Attn) │
        │  (MLP)                │
        │  (LayerNorm)          │
        └───────────┬───────────┘
                    │ (loop K times, K=1..8 at inference)
                    ▼
┌─────────────────────────────────────────────┐
│  Output Head: logits for all positions      │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  Energy Head (optional): scalar score       │  ← Contrastive validity score
└─────────────────────────────────────────────┘
```

### Key Differences from Phase 30

| Component | Phase 30 | Phase 31 |
|-----------|----------|----------|
| Depth | 4 fixed layers | 1-2 layers × K iterations (recurrent) |
| Noise Input | Implicit (mask token) | Explicit noise-level embedding |
| Training | Uniform random masking | **Structured noise curriculum** |
| Energy Head | None | Contrastive head on final hidden state |
| Inference | 5-step Gibbs | 1-8 step iterative refinement |

---

## Training: Continuous Noise Curriculum

### Noise Levels (σ ∈ [0, 1])

| σ | Interpretation | Training Task |
|---|----------------|---------------|
| 0.0 | Clean input | Identity (copy through) |
| 0.1-0.3 | Light corruption | **Repair** (typos, swaps) |
| 0.4-0.6 | Medium corruption | **Partial generation** |
| 0.7-0.9 | Heavy masking | **Generation from scratch** |
| 1.0 | Full mask | **Unconditional generation** |

### Corruption Types (Mutation Engine)

Inspired by Phase 18's synthetic mutations:
1. **Token replacement**: Random token → random other token
2. **Token deletion**: Remove token, shift left
3. **Token insertion**: Insert random token, shift right
4. **Swap**: Adjacent token swap
5. **Masking**: Replace with `<MASK>`

Each training sample: `(x_clean, x_corrupted, σ)` where `σ` controls corruption intensity.

### Loss Function

```python
# Primary: Reconstruction loss
loss_recon = cross_entropy(logits, x_clean, ignore_index=PAD)

# Secondary: Energy loss (contrastive)
# Positive pairs: (x_corrupted, x_clean) → energy ≈ 0
# Negative pairs: (x_corrupted, x_wrong) → energy ≈ 1
loss_energy = binary_cross_entropy(energy_head(hidden), labels)

# Combined
loss = loss_recon + λ * loss_energy  # λ = 0.1
```

---

## Implementation Plan

### Step 1: Extend Phase 30 Model (`model.py`)

```python
class UniversalDenoiser(nn.Module):
    def __init__(self, config):
        # Token + Position embeddings (from Phase 30)
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)

        # NEW: Noise level embedding
        self.wne = nn.Embedding(num_noise_levels, n_embd)  # or continuous via MLP

        # Recurrent block (single block, applied K times)
        self.block = Block(config, causal=False)

        # Output head
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Energy head
        self.energy_head = nn.Sequential(
            nn.Linear(n_embd, n_embd // 2),
            nn.ReLU(),
            nn.Linear(n_embd // 2, 1),
            nn.Sigmoid()
        )
```

### Step 2: Build Mutation Engine (`mutations.py`)

```python
def corrupt(x_clean, sigma, vocab_size, special_tokens):
    """Apply corruption at level sigma ∈ [0, 1]"""
    # Number of positions to corrupt
    n_corrupt = int(sigma * len(x_clean))

    # Sample corruption types
    for _ in range(n_corrupt):
        op = random.choice(['replace', 'delete', 'insert', 'swap', 'mask'])
        apply_mutation(x, op, vocab_size, special_tokens)

    return x_corrupted
```

### Step 3: Training Loop (`train_universal.py`)

```python
for batch in dataloader:
    x_clean = batch

    # Sample noise level
    sigma = torch.rand(batch_size) * 0.9 + 0.1  # [0.1, 1.0]

    # Corrupt
    x_corrupted = corrupt(x_clean, sigma, vocab_size, special_tokens)

    # Forward (K iterations at training time, K=1 or 2)
    hidden = embed(x_corrupted) + noise_embed(sigma)
    for _ in range(K_train):
        hidden = block(hidden)

    logits = lm_head(hidden)
    energy = energy_head(hidden.mean(dim=1))  # pool over sequence

    # Losses
    loss_recon = F.cross_entropy(logits, x_clean, ignore_index=PAD)
    loss_energy = contrastive_loss(energy, ...)

    loss = loss_recon + 0.1 * loss_energy
    loss.backward()
```

### Step 4: Inference Modes (`inference.py`)

```python
def generate(model, length, K_steps=5):
    """Generation: start from full mask"""
    x = torch.full((1, length), MASK_ID)
    for k in range(K_steps):
        logits = model(x, sigma=1.0 - k/K_steps)
        x = sample_or_argmax(logits)
    return x

def repair(model, x_corrupted, K_steps=3):
    """Repair: start from corrupted input"""
    x = x_corrupted.clone()
    for k in range(K_steps):
        logits = model(x, sigma=0.3)  # light noise assumption
        x = sample_or_argmax(logits)
    return x

def edit(model, x, mask_positions, K_steps=2):
    """Edit: mask specific positions, keep rest anchored"""
    x[mask_positions] = MASK_ID
    for k in range(K_steps):
        logits = model(x, sigma=0.5)
        x_new = sample_or_argmax(logits)
        x_new[~mask_positions] = x[~mask_positions]  # ANCHOR
        x = x_new
    return x
```

### Step 5: Evaluation (`benchmark.py`)

| Task | Metric | Target |
|------|--------|--------|
| Generation | Structural accuracy (vs ground truth syntax) | >85% |
| Repair | Edit distance reduction (corrupted → fixed) | >90% recovery |
| Editing | Anchor stability (unmasked positions unchanged) | 100% |
| Energy | ROC-AUC on valid vs invalid completions | >0.95 |

---

## File Structure

```
research-log/phase31-universal-denoiser/
├── PLAN_PHASE31.md          # This plan (copy from here)
├── model.py                 # UniversalDenoiser architecture
├── mutations.py             # Corruption/mutation engine
├── data.py                  # Dataset with (clean, corrupted, sigma) triples
├── train_universal.py       # Training loop
├── inference.py             # generate(), repair(), edit()
├── benchmark.py             # Evaluation harness
├── test_unified.py          # Unit tests for all three modes
├── RESULTS.md               # Final results (create after experiments)
└── checkpoints/             # Saved models
```

---

## Critical Files to Reuse

| Source | Purpose |
|--------|---------|
| `phase30/model.py:33-57` | BidirectionalSelfAttention (copy verbatim) |
| `phase30/model.py:72-86` | Block class (extend with recurrence) |
| `phase30/fractal_data.py` | FractalMathDataset (extend with mutations) |
| `phase30/flash_flood.py` | Gibbs sampling loop (adapt for K-step refinement) |
| `phase14/train_critic.py` | Contrastive energy training pattern |

---

## Success Criteria

**Minimum Viable Result:**
- Same model achieves >80% on generation AND >90% on repair AND 100% anchor stability
- This proves the "unified denoising" thesis

**Stretch Goal:**
- Beat Phase 30's 80.6% → reach 90%+ structural accuracy on generation
- Energy head correctly rejects >95% of invalid completions

**Negative Result (Informative Failure):**
- If recurrence doesn't help: "depth sharing is not the key"
- If energy head fails: "verification needs execution, not learned scores"
- If repair mode fails: "generation and repair are not the same task"

---

## Decisions

### Domain Progression
**Phase 31a:** Recursive arithmetic (proven ground from Phase 30)
**Phase 31b:** JSON/YAML configs (practical, structured)
**Phase 32:** Simple Python functions (ties to execution-based verification)

### Energy Head: Staged Training
1. **Stage 1:** Train bidirectional denoiser alone → strong reconstruction/parse accuracy (no energy loss)
2. **Stage 2:** Freeze trunk, add energy head, train on (corrupt, clean) pairs
3. **Stage 3 (optional):** Short joint fine-tune with λ_energy ≪ λ_recon if coupling is needed

This de-risks the architecture: if Stage 1 fails, we know denoising is the problem. If Stage 2 fails, we know energy scoring is the problem.

---

## Execution Order

```
Week 1: Stage 1 on Arithmetic
├── model.py (UniversalDenoiser, no energy head yet)
├── mutations.py (corruption engine)
├── data.py (arithmetic + mutations)
├── train_universal.py (reconstruction only)
└── benchmark.py (generation, repair, edit metrics)

Week 2: Add Energy + Move to JSON
├── Add energy_head to model.py
├── train_energy.py (frozen trunk)
├── json_data.py (JSON dataset + mutations)
├── Repeat benchmarks on JSON domain
└── RESULTS.md

Phase 32 (future): Python functions + execution
```
