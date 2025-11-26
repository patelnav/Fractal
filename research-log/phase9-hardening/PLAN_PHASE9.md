# Phase 9: Adversarial Hardening

**Objective:** Improve the Ouroboros Verifier by training on "Hard Negatives" — incorrect solutions that the current model confidently mistakes for correct ones.

**Context:**
In Phase 8, we achieved a **+19.4% accuracy boost** (36% -> 43%) primarily by filtering out malformed answers. However, the model still struggles to distinguish subtle reasoning errors from correct logic, often assigning low energy (0.001) to plausible-sounding but wrong hallucinations.

To reach >90% accuracy (System 2 reliability), the verifier must be exposed to these "Wolves in Sheep's Clothing."

---

## The Mission

You are to execute the **Adversarial Hardening Loop**:
1.  **Benchmark:** Verify generation speed is SOTA (>50 seq/sec) on the A100. Do not proceed if slow.
2.  **Mine:** Generate candidates for the entire **GSM8K Training Set (7.5k problems)**.
    *   Filter for: `(Model says Correct) AND (Ground Truth says Wrong)`.
    *   Target: Collect ~5,000 - 10,000 hard negatives.
3.  **Retrain:** Fine-tune the Phase 7 checkpoint (`checkpoints/ckpt.pt`) on this new dataset mixed with the original Phase 7 data.

---

## Step 1: Speed Benchmark (Mandatory)

Before running the full mining job (~15 mins), you must verify the "Turbo Mode" parallelization is working.
Create `benchmark_generation.py`:
*   Load `generator.py` (HuggingFaceGenerator).
*   Generate 5 candidates for 100 dummy prompts.
*   **Pass Criteria:** Total time < 15 seconds.
*   **Fail Action:** If slow, fix `generator.py` batching logic (ensure `generate_batch` uses single forward pass).

## Step 2: Mine Negatives

Create `mine_negatives.py` (based on `phase8-solver/solve_math.py`):
*   Load GSM8K **TRAIN** set (`data/gsm8k/train.json`).
*   Generate 5 candidates per problem.
*   Score with `OuroborosModel`.
*   Save ONLY the failures to `data/hard_negatives.jsonl`:
    ```json
    {
        "question": "...",
        "wrong_solution": "...",
        "energy_score": 0.002,  # The low score we want to penalize
        "ground_truth": "..."
    }
    ```

## Step 3: Retrain

Create `train_hardening.py` (based on `phase7-ouroboros/train.py`):
*   Load Phase 7 Checkpoint.
*   Load Original Phase 7 Data (`data/processed/train.npz`).
*   Load Hard Negatives.
*   **Loss Function:** Contrastive.
    *   Original Data: Keep maintaining Energy(Correct) -> 0.
    *   Hard Negatives: Push Energy(Hard Negative) -> 1.
*   Train for ~1000 iterations (fast fine-tuning).

---

## Environment & Resources

**Lambda Labs A100:**
*   IP: (Use `lambda_helper.py status` to find)
*   Path: `~/Fractal/research-log/phase9-hardening/`
*   Model Checkpoint: `~/Fractal/research-log/phase7-ouroboros/checkpoints/ckpt.pt`

**Helper Commands:**
```bash
# Check status
python lambda_helper.py status

# Sync code
python lambda_helper.py sync

# Run benchmark
ssh lambda "cd ~/Fractal/research-log/phase9-hardening && python benchmark_generation.py"
```

**Key Constraints:**
*   **DO NOT** touch the GSM8K **Test Set** for training/mining. Only use `train.json`.
*   **DO NOT** overwrite the Phase 7 checkpoint. Save new model as `checkpoints/ckpt_hardened.pt`.
