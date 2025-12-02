# Next Steps - Phase 3: Training & Evaluation

**Status:** Phase 1-2 COMPLETE | **Now:** Phase 3
**Goal:** Train 3 models on full dataset, evaluate, select best, integrate into DiariZen

---

## What We Have (from Phase 1-2)

| Model | Params | Overfit MAE | Status |
|-------|--------|-------------|--------|
| **Transformer** | 1.4M | **3ms** | Primary candidate |
| MLP | 822K | 25ms | Baseline |
| Contrastive | 361K | 40ms | Alternative |

**Baseline to beat:** DiariZen DER = ~9.1%, boundary error = 923ms average
**Target:** <100ms boundary MAE (10x improvement), DER < 8%

---

## Phase 3 Action Items

### 0. Lambda Session: Re-baseline + Training

**On Lambda A100:**

```bash
# 1. Setup
cd ~/diarize/DiariZen
source .venv/bin/activate  # or conda activate diarizen

# 2. Re-run evaluation (fix already applied - no --ignore_overlaps)
python evaluate_voxconverse.py 2>&1 | tee /tmp/eval_baseline.txt
# Expected: DER ~9.1% (was 4.52% with bug)

# 3. Launch training (all 3 in parallel)
cd ../boundary_refinement
tmux new-session -d -s train_transformer 'python scripts/train.py --config configs/transformer.toml 2>&1 | tee /tmp/train_transformer.txt'
tmux new-session -d -s train_mlp 'python scripts/train.py --config configs/mlp.toml 2>&1 | tee /tmp/train_mlp.txt'
tmux new-session -d -s train_contrastive 'python scripts/train.py --config configs/contrastive.toml 2>&1 | tee /tmp/train_contrastive.txt'

# 4. Monitor
tmux ls  # See running sessions
tmux attach -t train_transformer  # Watch transformer (Ctrl+B D to detach)
tail -f /tmp/train_*.txt  # Or watch all logs
```

**Zone C:** ~5-6 hours GPU time (parallel)
**Cost:** ~$7-8

### 1. Launch Training (~1 decision point)

```bash
# On Lambda A100
cd diarize/boundary_refinement

# Run all 3 in parallel
python scripts/train.py --config configs/transformer.toml &
python scripts/train.py --config configs/mlp.toml &
python scripts/train.py --config configs/contrastive.toml &
```

**Config decisions:**
- Dataset: VoxConverse train (~5k boundaries + 2x synthetic = ~15k examples)
- Epochs: 50 (Transformer), 50 (MLP), 50 (Contrastive)
- Batch size: 32
- Learning rate: 5e-4

### 2. Monitor Training (~2-3 decision points, Zone C: 5-6 hours)

- Check TensorBoard every 1-2 hours
- Watch for: loss decreasing, no NaN/Inf, validation MAE improving
- GPU time: ~5 hours per model (parallel = ~6 hours wall clock)

### 3. Evaluate Results (~2 decision points)

After training completes:
```bash
python scripts/evaluate.py --model transformer --checkpoint best.pt
python scripts/evaluate.py --model mlp --checkpoint best.pt
python scripts/evaluate.py --model contrastive --checkpoint best.pt
```

**Metrics to compare:**
- Boundary MAE (target: <100ms)
- Accuracy @50ms, @100ms, @200ms
- Inference speed

### 4. Integrate Best Model (~2 decision points)

- Plug best model into DiariZen pipeline
- Run end-to-end on VoxConverse dev
- Measure DER improvement (target: 4.52% → <3%)

### 5. Decision Gate 4

- [ ] Best model achieves <100ms MAE?
- [ ] DER improves by >10%?
- [ ] Worth iterating or ready for Phase 4?

---

## Estimate

| Resource | Amount |
|----------|--------|
| Decision points | ~10 |
| Zone C (GPU) | ~6-8 hours |
| Cost | ~$8-10 |
| Wall clock | 1-2 days |

**Engagement:** Intermittent (check training hourly, decisions in minutes)

---

## Quick Reference

**Budget:** $999.28 remaining
**LOG.md:** Historical record (Phase 1-2 details)
**PLAN.md:** Original 14-day plan + decision gates

**Ping:** "Start training" or "Launch Phase 3"
