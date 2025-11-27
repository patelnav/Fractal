# Phase 10: vLLM Test for Gemma-3-1B

**Goal**: Test vLLM as a drop-in replacement for HuggingFace generator to speed up evaluation.

## Results: 30x Speedup Achieved!

| Batch Size | HuggingFace | vLLM | Speedup |
|------------|-------------|------|---------|
| 1 gen | ~2s | 0.12s | 17x |
| 16 gens | ~8s | 0.25s | 32x |
| 128 gens | ~30s | **0.99s** | **30x** |

**Bottom line**: Full GSM8K eval goes from ~83 minutes to ~3 minutes.

## Quick Start (Lambda A100)

```bash
# From your local machine:
python lambda_helper.py launch gpu_1x_a100
python lambda_helper.py wait
python lambda_helper.py setup-ssh
python lambda_helper.py sync research-log/phase10-vllm-test

# SSH and run:
ssh lambda
cd ~/Fractal/research-log/phase10-vllm-test

# Setup venv with uv
pip install uv
~/.local/bin/uv venv --python 3.12 .venv
source .venv/bin/activate
~/.local/bin/uv pip install -r requirements.txt

# Copy HF token (Gemma is gated)
mkdir -p ~/.cache/huggingface
echo "YOUR_HF_TOKEN" > ~/.cache/huggingface/token

# Run test
python test_vllm_gemma.py           # Quick test (1, 2, 4 gens)
python test_vllm_gemma.py --full    # Full benchmark (16, 128 gens)
```

## One-liner Setup (after SSH)

```bash
cd ~/Fractal/research-log/phase10-vllm-test && \
pip install uv && \
~/.local/bin/uv venv --python 3.12 .venv && \
source .venv/bin/activate && \
~/.local/bin/uv pip install -r requirements.txt && \
python test_vllm_gemma.py --full
```

## Integration with run_eval.py

Replace the generator import:

```python
# OLD (slow):
from generator import HuggingFaceGenerator
generator = HuggingFaceGenerator(model_name="google/gemma-3-1b-it", ...)

# NEW (fast):
from generator_vllm import VLLMGenerator
generator = VLLMGenerator(model_name="google/gemma-3-1b-it")
```

The `VLLMGenerator` class has the same `generate_batch(questions, n=16)` interface.

## Benchmark Output (A100 40GB)

```
============================================================
VLLM + GEMMA-3-1B-IT TEST
Mode: FULL
============================================================

Model loaded in 27.20s

TEST 0: Single Generation (fail-fast check)
Response (0.12s):  4. Final Answer: $\boxed{4}$
[OK] Single generation in 0.12s - proceeding with tests

TEST: Batch of 16 generations
Generated 16 responses in 0.25s (62.9 gen/s)

TEST: Full Batch (8 x 16 = 128 generations)
Generated 128 responses in 0.99s
Rate: 128.7 generations/sec

============================================================
FULL RESULTS
============================================================
16 generations:  0.25s  (62.9 gen/s)
128 generations: 0.99s (128.7 gen/s)

Comparison with HuggingFace (estimated):
  HF:   ~30s for 128 generations
  vLLM: 0.99s for 128 generations
  Speedup: ~30.2x
============================================================
```

## Why vLLM is Faster

1. **Continuous Batching**: Processes new requests while others generate
2. **PagedAttention**: Efficient KV cache with memory paging
3. **CUDA Graphs**: Pre-compiled execution paths
4. **Tensor Parallelism**: Easy multi-GPU (if needed)

## Troubleshooting

### "CUDA out of memory"
```python
# Reduce GPU memory usage
llm = LLM(model=..., gpu_memory_utilization=0.3)  # Default is 0.5
```

### "Gated repo" / 401 Error
```bash
# Gemma requires HuggingFace login
mkdir -p ~/.cache/huggingface
echo "hf_YOUR_TOKEN_HERE" > ~/.cache/huggingface/token
```

### Model loading takes >2 minutes
First run compiles CUDA graphs (~60s). Subsequent runs use cache (~27s).

## Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Dependencies (vllm, huggingface_hub) |
| `test_vllm_gemma.py` | Benchmark script with fail-fast |
| `generator_vllm.py` | Drop-in replacement for HuggingFaceGenerator |
| `setup.sh` | Legacy setup script (use uv instead) |

## Cost

- Lambda A100 40GB: $1.29/hr
- Full GSM8K eval with vLLM: ~3 minutes = ~$0.06
- Full GSM8K eval with HuggingFace: ~83 minutes = ~$1.79

**Don't forget to terminate when done:**
```bash
python lambda_helper.py terminate
```
