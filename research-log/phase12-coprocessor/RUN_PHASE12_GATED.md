
# Running Phase 12.1: Gated Fractal Multiplier

## 1. Sync
```bash
python lambda_helper.py sync research-log
```

## 2. Retrain Adder (Remote)
Since we changed the architecture (Added GRU Gating), the old checkpoint is invalid. We must retrain the Adder first.
```bash
ssh lambda
cd ~/Fractal/research-log/phase11-fractal-logic
source .venv/bin/activate

# Retrain Adder (Should be fast, 100% acc expected)
python train_logic.py
```

## 3. Train Multiplier (Remote)
Once the Adder is retrained (checkpoint saved), we train the Multiplier.
```bash
cd ~/Fractal/research-log/phase12-coprocessor

# Update checkpoint path if needed in train_mult.py (it points to ckpt_e10.pt by default)
# We assume train_logic.py overwrites the old checkpoints.

# Train Multiplier
python train_mult.py
```

## 4. Hypothesis
The **GRU Gating** provides a linear path for gradients (via $1-z_t$), solving the vanishing gradient problem in the deep recursion ($O(N^2)$).
This should allow the Multiplier to learn the shift-and-add loop.
