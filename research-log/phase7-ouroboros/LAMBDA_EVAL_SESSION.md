# GSM8K Evaluation Session - Lambda Labs A100

**Started:** 2025-11-26
**Instance ID:** `ae87d2b3777f44cd89c8215641893c54`
**Instance Type:** `gpu_1x_a100_sxm4` @ $1.29/hr
**Region:** us-west-2

## Current Status

**Evaluation V2 RUNNING** - Fixed version with chat template support.

### Log File
```bash
ssh lambda "tail -f ~/Fractal/research-log/phase7-ouroboros/gsm8k_eval_v2.log"
```

### Critical Fixes Applied (V2)
1. **Chat Template Fix**: generator.py now uses `tokenizer.apply_chat_template()` for proper instruction-tuned model formatting
2. **Incremental Saving**: solve_math.py saves results to `results_incremental.jsonl` after each batch (crash-safe)

### Previous Run (V1) - INVALID
- Processed: 256/1319 problems (~19%) before crash
- Baseline: 36.1% (should be ~60% with proper prompting)
- **Issue**: Raw prompts were passed without chat template formatting

## How to Resume/Monitor

### 1. Check Instance Status
```bash
cd ~/Developer/fractal
python lambda_helper.py status
```

### 2. Setup SSH (if needed)
```bash
python lambda_helper.py setup-ssh
```

### 3. Monitor Evaluation
```bash
# Quick check
ssh lambda "tail -30 ~/Fractal/research-log/phase7-ouroboros/gsm8k_eval_v2.log"

# Continuous monitoring
ssh lambda "tail -f ~/Fractal/research-log/phase7-ouroboros/gsm8k_eval_v2.log"

# Check if still running
ssh lambda "pgrep -af solve_math"

# Check incremental results (crash-safe)
ssh lambda "wc -l ~/Fractal/research-log/phase7-ouroboros/results/gsm8k_eval/results_incremental.jsonl"
```

### 4. Download Results (when complete)
```bash
# Results will be in:
scp lambda:~/Fractal/research-log/phase7-ouroboros/results/gsm8k_eval/* ./results/gsm8k_eval/

# Download incremental results (available even if crashed)
scp lambda:~/Fractal/research-log/phase7-ouroboros/results/gsm8k_eval/results_incremental.jsonl ./

# Download full log
scp lambda:~/Fractal/research-log/phase7-ouroboros/gsm8k_eval_v2.log ./
```

### 5. Terminate Instance (when done)
```bash
python lambda_helper.py terminate
```

## Code Fixes Applied (V2)

### 1. Chat Template for Instruction-Tuned Models
**Problem:** generator.py passed raw text prompts to instruction-tuned models, which expect formatted input like:
```
<start_of_turn>user
Solve this problem...
<end_of_turn>
<start_of_turn>model
```

**Fix in generator.py:**
```python
# In HuggingFaceGenerator.generate():
messages = [{"role": "user", "content": prompt}]
inputs = self.tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(self.device)
```

### 2. Incremental Results Saving
**Problem:** solve_math.py only saved results at the end. If the process crashed, all progress was lost.

**Fix in solve_math.py:**
```python
# Save to JSONL after each batch
with open(jsonl_path, 'a') as f:
    for result in batch_results:
        f.write(json.dumps(asdict(result)) + '\n')
```

## Dependency Fixes Applied

The following issues were encountered and fixed:

1. **NumPy 2.x compatibility**: vLLM installed numpy 2.2.6 which broke system packages
   - Fixed: `pip install --upgrade scipy scikit-learn pandas ml-dtypes`
   - Fixed: `pip install tensorflow-cpu` (user-space version shadows system)

2. **Checkpoint transfer**: rsync excluded *.pt files
   - Fixed: Manually scp'd the 729MB checkpoint

3. **HuggingFace token**: Required for gated Gemma-3 model
   - Set via: `export HF_TOKEN=hf_...`

## Configuration

**Generator:** HuggingFace (batched generation)
**Model:** google/gemma-3-1b-it (1B params, 62.8% GSM8K baseline)
**Batch size:** 16 problems x 5 candidates = 80 sequences per forward pass
**Temperature:** 0.7
**Top-p:** 0.9

## Files on Lambda

```
~/Fractal/research-log/phase7-ouroboros/
├── solve_math.py           # Main evaluation script (with incremental saving)
├── generator.py            # LLM generators with chat template support
├── utils.py                # Prompt formatting, answer parsing
├── checkpoints/ckpt.pt     # Ouroboros model (729MB)
├── data/gsm8k/test.json    # GSM8K test set (1,319 problems)
├── config/eval_gsm8k.py    # Evaluation config
├── gsm8k_eval_v2.log       # Current evaluation log (V2 with fixes)
├── gsm8k_eval_final.log    # Previous evaluation log (V1 - invalid)
└── results/gsm8k_eval/
    ├── results.json            # Final results (on completion)
    └── results_incremental.jsonl # Per-batch results (crash-safe)
```

## Lambda Helper Commands

```bash
# From ~/Developer/fractal:
python lambda_helper.py launch    # Launch new instance
python lambda_helper.py wait      # Wait for instance to be ready
python lambda_helper.py status    # Show instance status and IP
python lambda_helper.py setup-ssh # Configure SSH for 'ssh lambda'
python lambda_helper.py sync      # Sync code to instance
python lambda_helper.py run "cmd" # Run command on instance
python lambda_helper.py terminate # Terminate instance
```

## Expected Results

### V2 (with chat template - current run)
- **Baseline (random):** ~60% (gemma-3-1b-it expected accuracy with proper prompting)
- **Ouroboros (energy):** Should improve over baseline if energy model is discriminating
- **Oracle (best of 5):** ~75-80% (upper bound with perfect selection)

### V1 (without chat template - INVALID)
- **Baseline (random):** 36.1% (low because model wasn't prompted correctly)
- **Oracle:** 58.7% (still had some correct answers by chance)
- **Why invalid:** Instruction-tuned models require chat template formatting (e.g., `<start_of_turn>user...`) to perform correctly

The gap between Oracle and Baseline represents the potential improvement from good reranking.
