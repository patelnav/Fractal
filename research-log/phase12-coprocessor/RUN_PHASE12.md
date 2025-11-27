# Running Phase 12: Differentiable Multiplier Training

## 1. Sync
```bash
python lambda_helper.py sync research-log/phase12-coprocessor
```

## 2. Train (Remote)
The Zero-Shot attempt failed because the "Soft Accumulator" (hidden states) drifts when looping without training. We now "Close the Loop" by training the Adder to handle its own output recursively.

```bash
ssh lambda
cd ~/Fractal/research-log/phase12-coprocessor
source .venv/bin/activate

# Ensure Adder checkpoint exists
ls ../phase11-fractal-logic/checkpoints_recurrent/ckpt_e10.pt

# Train
python train_mult.py
```

## 3. Expected Result
*   **Train Acc:** Should hit 100% (6-bit multiplication).
*   **Test Acc:** If the Fractal Hypothesis holds (Compositionality), it should extrapolate to 8-bit multiplication ($>90\%$).
*   This would prove we can build **Differentiable Neural Programs** that generalize.