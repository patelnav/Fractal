# Phase 15 GRPO - Runner Context

## Current Status
An H100 80GB PCIe instance is **currently booting** on Lambda Labs.
- Instance ID: `ace5f8916191454da0cbe1520b337a9b`
- IP: `209.20.158.84`
- Cost: $2.49/hr
- Status: Still "booting" as of last check

## What Happened
1. Phase 14 GRPO training on A100 40GB failed with **CUDA OOM** even at minimal batch sizes (2 prompts x 4 samples = 8 sequences)
2. Root cause: Two 1.5B models (policy + ref) + forward passes requires more than 40GB VRAM
3. Launched H100 80GB to handle aggressive batch sizes

## train_grpo.py - Updated Settings
The script has been updated for H100:
```python
GROUP_SIZE = 8 # G - samples per prompt
MINI_BATCH_SIZE = 16 # Prompts per step (Total batch = 16 * 8 = 128 sequences)
GRAD_ACCUM = 1 # H100 80GB can handle full batch
```

## Next Steps for New Thread
1. **Wait for H100 to be ready**: `python lambda_helper.py wait`
2. **Sync the updated phase15-rl**: `python lambda_helper.py sync research-log/phase15-rl`
3. **Run GRPO training**:
   ```bash
   ssh lambda "source ~/.local/bin/env && cd ~/Fractal && source research-log/phase11-fractal-logic/.venv/bin/activate && python research-log/phase15-rl/train_grpo.py" 2>&1 | tee /tmp/phase15_grpo.log
   ```
4. Monitor reward improvement: Should see `Reward=X.XX` in log output

## Key Learnings

### Memory Management
- Qwen 1.5B needs ~3GB per copy in bfloat16
- Two models (policy + ref) = ~6GB baseline
- Forward pass on 128 sequences + 512 tokens = significant activation memory
- Gradient checkpointing helps but not enough for A100 40GB with GRPO's requirements
- H100 80GB is the right choice for this workload

### Code Fixes Made
1. `torch_dtype` deprecation warning is cosmetic (works anyway)
2. bfloat16 → float() conversion needed before `.cpu().numpy()` (already fixed in test scripts)

### GRPO Algorithm Notes
- Uses group-relative advantages: (reward - group_mean) / group_std
- KL penalty: `log_pi - log_ref` term prevents divergence from base model
- Policy gradient: `-advantages * log_probs` per token, masked to generation only

## Phase 14 Results (for reference)
- Critic Pass@1 improvement: +5.98% on MBPP test set
- HumanEval transfer: +3.67% improvement (64.62% → 68.29%)
- Hardened critic performed worse than original
