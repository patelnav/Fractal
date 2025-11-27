
# Running Phase 13: The Neural RISC-V

## 1. Sync
```bash
python lambda_helper.py sync research-log/phase13-neural-riscv
```

## 2. Generate Data (Remote)
```bash
ssh lambda
cd ~/Fractal/research-log/phase13-neural-riscv
python generate_riscv_data.py
```

## 3. Train the Controller (Remote)
We are training a Transformer Controller to output instructions (`ADD`, `SHIFT`) that manipulate a set of registers to transform `A` and `B` into `A*B`.
The "ALU" (Adder) is frozen. The gradients flow through the instructions (via Gumbel-Softmax) to update the Controller's policy.

```bash
source .venv/bin/activate

# Ensure Phase 11 checkpoint is present
ls ../phase11-fractal-logic/checkpoints_recurrent/ckpt_e10.pt

# Train
python train_riscv.py
```

## 4. Success Criteria
*   If the model reaches **100% Training Accuracy**, it has discovered an algorithm.
*   If it reaches **>90% Test Accuracy** (Extrapolation), it has discovered the *correct* algorithm (Shift-and-Add) which generalizes to any bit width.
