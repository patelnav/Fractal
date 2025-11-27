# Fractal Project: Session Context & Lambda Runner Guide

This document provides context for resuming work across sessions. **Read this first when starting a new thread.**

---

## Current Phase

| Field | Value |
|-------|-------|
| **Active Phase** | `phase14-vector6-reboot` |
| **Phase Goal** | Best-of-N code selection with learned critic |
| **Status** | COMPLETE |

---

## Phase 14 Final Results

| Step | Status | Output |
|------|--------|--------|
| 1. Generate MBPP Train | COMPLETE | 8,150 samples in 90s (vLLM) |
| 2. Execute/Label Train | COMPLETE | 63.19% pass rate (5,150/8,150) |
| 3. Train Critic | COMPLETE | Val Acc = 88.83% (exceeds 80% target) |
| 4. Generate MBPP Test | COMPLETE | 12,850 samples (257 problems x 50) |
| 5. Execute/Label Test | COMPLETE | 60.94% pass rate (7,831/12,850) |
| 6. Critic Ranking | **COMPLETE** | **+5.98% improvement** |

### Key Metrics
```
Baseline Pass@1 (Random): 60.94%
Critic   Pass@1 (Top-1):  66.93%
Oracle   Pass@N (Upper):  90.27%
Improvement: +5.98% absolute (~10% relative)
```

### Phase 14 Files on Lambda
```
~/Fractal/research-log/phase14-vector6-reboot/
├── generate_mbpp.py          # vLLM generation (train)
├── generate_mbpp_test.py     # vLLM generation (test)
├── execute_mbpp.py           # Hard Verifier (train)
├── execute_mbpp_test.py      # Hard Verifier (test)
├── train_critic.py           # Soft Verifier training
├── test_critic_ranking.py    # Best-of-N evaluation
├── data/
│   ├── mbpp_generations.jsonl     # 8,150 train samples
│   ├── mbpp_labeled.jsonl         # Train with pass/fail labels
│   ├── mbpp_test_generations.jsonl # 12,850 test samples
│   └── mbpp_test_labeled.jsonl    # Test with pass/fail labels
└── checkpoints_critic/            # critic_e1.pt, critic_e2.pt, etc.
```

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
python lambda_helper.py sync research-log/PHASE_DIR
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

## Key Context Files

| File | Purpose |
|------|---------|
| `RUNNER_CONTEXT.md` | This file - session context & Lambda guide |
| `research-log/LOG.md` | Historical results and learnings |
| `research-log/PHASE_DIR/README.md` | Phase-specific technical details |
| `research-log/PHASE_DIR/RUN_PHASE*.md` | Success criteria and steps |
| `CLAUDE.md` | Project instructions for Claude |

---

## Running a New Experiment: Checklist

1. **Read this file** - Check current phase status
2. **Read the plan** - Check `RUN_PHASE*.md` for success criteria
3. **Sync code** - `python lambda_helper.py sync research-log/PHASE_DIR`
4. **Run script with logging** - Always tee output to `/tmp/`
5. **Monitor progress** - `tail -f /tmp/LOGFILE.log`
6. **Check results** - Grep for key metrics
7. **Update this file** - Record results in Current Phase section

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

## Architecture Summary (Phase 14)

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
Trained Critic Model (88.83% val acc)
    ↓
Best-of-N Selection (rank by critic score, pick highest)
    ↓
Result: 60.94% → 66.93% Pass@1 (+5.98%)
```

---

## Lambda Costs

- A100 SXM4 80GB: ~$1.29/hr
- Always terminate when done: `python lambda_helper.py terminate`

---

## Next Steps / Future Work

- [ ] Scale to larger models (7B, 14B)
- [ ] Try iterative refinement with critic feedback
- [ ] Explore Best-of-N with higher N
- [ ] Test on other benchmarks (HumanEval, APPS)
