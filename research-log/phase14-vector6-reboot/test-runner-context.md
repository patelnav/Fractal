# Fractal Project: Lambda Test Runner Guide

This document provides context for running ML experiments on Lambda GPU instances. Read this first when resuming work.

---

## Quick Reference

| Item | Value |
|------|-------|
| Lambda SSH Alias | `lambda` (configured via `lambda_helper.py setup-ssh`) |
| Shared venv | `research-log/phase11-fractal-logic/.venv` |
| Key packages | vLLM, transformers, torch, datasets |
| Remote project root | `~/Fractal/` |

---

## Essential Lambda Commands

### 1. Instance Lifecycle

```bash
# Launch a new instance
python lambda_helper.py launch

# Wait for instance to be ready
python lambda_helper.py wait

# Setup SSH config (run after wait, or when IP changes)
python lambda_helper.py setup-ssh

# Terminate instance
python lambda_helper.py terminate
```

### 2. Sync Local Code to Lambda

```bash
# Sync a specific phase directory
python lambda_helper.py sync research-log/phase14-vector6-reboot
```

### 3. Run Scripts on Lambda

**Standard template (ALWAYS use this pattern):**
```bash
ssh lambda "source ~/.local/bin/env && cd ~/Fractal && \
  source research-log/phase11-fractal-logic/.venv/bin/activate && \
  python research-log/PHASE_DIR/SCRIPT.py" 2>&1 | tee /tmp/LOGFILE.log
```

**Why this pattern:**
- `source ~/.local/bin/env` - Loads uv/Python paths
- `cd ~/Fractal` - Sets working directory
- `source .venv/bin/activate` - Uses shared venv with all dependencies
- `2>&1 | tee /tmp/FILE.log` - Captures all output locally for review

### 4. Check Running Processes

```bash
# See what's running on Lambda
ssh lambda "ps aux | grep python | grep -v grep"

# Check GPU usage
ssh lambda "nvidia-smi"
```

---

## Key Context Files to Read

Before running any phase, read these files for context:

| File | Purpose |
|------|---------|
| `research-log/PHASE_DIR/RUN_PHASE*.md` | Success criteria and steps |
| `research-log/PHASE_DIR/README.md` | Technical details |
| `research-log/LOG.md` | Historical results and learnings |

---

## Running a New Experiment: Checklist

1. **Read the plan** - Check `RUN_PHASE*.md` for success criteria
2. **Sync code** - `python lambda_helper.py sync research-log/PHASE_DIR`
3. **Run script with logging** - Always tee output to `/tmp/`
4. **Monitor progress** - `tail -f /tmp/LOGFILE.log`
5. **Check results** - Grep for key metrics (e.g., `grep "Val Acc" /tmp/*.log`)
6. **Update status** - Edit this file with results

---

## Common Patterns

### vLLM Generation (Fast LLM Inference)
```bash
ssh lambda "source ~/.local/bin/env && cd ~/Fractal && \
  source research-log/phase11-fractal-logic/.venv/bin/activate && \
  python research-log/PHASE_DIR/generate_*.py" 2>&1 | tee /tmp/gen.log
```
- vLLM is 50-100x faster than HuggingFace transformers
- Use for generating many samples (n=50 per prompt)

### Training with Checkpoints
```bash
ssh lambda "source ~/.local/bin/env && cd ~/Fractal && \
  source research-log/phase11-fractal-logic/.venv/bin/activate && \
  python research-log/PHASE_DIR/train_*.py" 2>&1 | tee /tmp/train.log
```
- Training scripts save checkpoints to `checkpoints_*/` directories
- Look for "Val Acc", "Train Loss", or similar metrics in output

### Code Execution (Hard Verification)
```bash
ssh lambda "source ~/.local/bin/env && cd ~/Fractal && \
  source research-log/phase11-fractal-logic/.venv/bin/activate && \
  python research-log/PHASE_DIR/execute_*.py" 2>&1 | tee /tmp/exec.log
```
- Runs generated code against unit tests
- Reports pass/fail rates

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| SSH timeout | Run `python lambda_helper.py setup-ssh` again |
| vLLM not found | `ssh lambda "source ~/.local/bin/env && pip install vllm"` |
| CUDA OOM | Reduce batch size in training script |
| Process died | Check with `ps aux`, restart script |
| Instance terminated | Run `python lambda_helper.py launch && wait && setup-ssh` |

---

## Phase 14 Current Status

| Step | Status | Output |
|------|--------|--------|
| 1. Generate MBPP | COMPLETE | 8,150 samples in 90s (vLLM) |
| 2. Execute/Label | COMPLETE | 63.19% pass rate (5,150/8,150) |
| 3. Train Critic | **COMPLETE** | Val Acc = 88.83% (exceeds 80% target) |

### Phase 14 Files on Lambda

```
~/Fractal/research-log/phase14-vector6-reboot/
├── generate_mbpp.py      # vLLM-based generation (Qwen2.5-Coder-1.5B)
├── execute_mbpp.py       # Hard Verifier (runs tests, labels pass/fail)
├── train_critic.py       # Soft Verifier (Qwen + classification head)
├── data/
│   ├── mbpp_generations.jsonl  # 8,150 samples
│   └── mbpp_labeled.jsonl      # Same with status labels
└── checkpoints_critic/         # Model checkpoints (critic_e1.pt, etc.)
```

### Next Steps (Phase 14.5)

1. Generate test set samples (MBPP test split)
2. Score all samples with trained critic
3. Select highest-scoring sample per problem
4. Measure Pass@1 improvement vs random selection

---

## Architecture Summary

```
Generator (Qwen2.5-Coder-1.5B-Instruct)
    ↓
N samples per problem (e.g., n=50)
    ↓
Hard Verifier (execute code, run tests)
    ↓
Labeled data (passed/failed)
    ↓
Soft Verifier Training (learns to predict pass/fail)
    ↓
Trained Critic Model
    ↓
Best-of-N Selection (rank by critic score, pick highest)
```

---

## Lambda Costs

- A100 SXM4 80GB: ~$1.29/hr
- Always terminate when done: `python lambda_helper.py terminate`
