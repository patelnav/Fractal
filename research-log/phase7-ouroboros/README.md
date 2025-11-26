# Phase 7: Ouroboros Reasoner

Energy-based self-correction for math and code reasoning.

## Goal

Prove that an energy head can distinguish correct vs incorrect reasoning steps.

**Success Criteria:**
- Energy(correct) < 0.2
- Energy(wrong) > 0.8
- Detection accuracy > 90%

## Architecture

- **Model**: 12-layer transformer with RoPE (512 dim, 8 heads)
- **Energy Head**: 2-layer MLP (512 → 256 → 1)
- **Loss**: Contrastive MSE: E(correct) → 0, E(wrong) → 1
- **Data**: Balanced 50/50 correct/wrong pairs

## Files

```
phase7-ouroboros/
├── download_data.py      # Fetch GSM8K + HumanEval
├── tokenizer.py          # Tiktoken-based tokenizer
├── prepare_data.py       # Generate balanced contrastive pairs
├── model.py              # OuroborosModel with energy head
├── train.py              # nanoGPT-style training loop
├── evaluate.py           # Energy discrimination tests
├── config/
│   ├── train_m2.py       # Local testing (6 layers, 256 dim)
│   └── train_a100.py     # Full training (12 layers, 512 dim)
└── data/
    ├── gsm8k/            # Math problems
    ├── humaneval/        # Code problems
    └── processed/        # train.npz, val.npz
```

## Lambda Labs Training

### 1. Spin up A100 instance

Go to [Lambda Labs](https://lambdalabs.com) and launch a 1x A100 80GB instance (~$1.10/hr).

### 2. SSH and clone

```bash
ssh ubuntu@<instance-ip>
git clone https://github.com/yourname/fractal.git
cd fractal/research-log/phase7-ouroboros
```

### 3. Install dependencies

```bash
pip install torch numpy tiktoken tqdm wandb datasets
```

### 4. Download and prepare data

```bash
python download_data.py
python prepare_data.py
```

### 5. Train

```bash
# With wandb logging
wandb login
python train.py config/train_a100.py

# Or without wandb
python train.py config/train_a100.py --wandb_log=False
```

Training should take ~2-3 hours on A100.

### 6. Evaluate

```bash
python evaluate.py --checkpoint checkpoints/ckpt.pt --device cuda
```

### 7. Download results

```bash
scp ubuntu@<ip>:fractal/research-log/phase7-ouroboros/checkpoints/ckpt.pt .
```

## Expected Results

After ~5000 iterations on A100:

| Metric | Target | Phase 3-4 Achieved |
|--------|--------|-------------------|
| Energy correct | < 0.2 | 0.009 |
| Energy wrong | > 0.8 | 0.984 |
| Accuracy | > 90% | 100% |
| Separation | > 0.6 | 0.975 |

Phase 3-4 achieved 100% on Shakespeare BPE decompression. Ouroboros extends this to math/code reasoning.

## Cost Estimate

| Phase | Time | Cost |
|-------|------|------|
| Data prep (local) | 5 min | $0 |
| Training (A100) | 3 hrs | ~$3.30 |
| Eval (A100) | 10 min | ~$0.20 |
| **Total** | **~3.5 hrs** | **~$4** |
