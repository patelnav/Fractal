
# Running Phase 12.2: Digital Restoration (Gumbel-Softmax)

## 1. Sync
```bash
python lambda_helper.py sync research-log/phase12-coprocessor
```

## 2. Train (Remote)
We have modified `model_fractal_mult.py` to use **Hard Gumbel-Softmax**. This forces the accumulator to snap to a valid embedding ("0" or "1") at every step, preventing drift, while allowing gradients to flow via the Straight-Through Estimator.

```bash
ssh lambda
cd ~/Fractal/research-log/phase12-coprocessor
source .venv/bin/activate

# Train Multiplier (using same Adder checkpoint)
python train_mult.py
```

## 3. Hypothesis
*   **Previous Failure:** Analog Drift (Soft embeddings became noise after 16 steps).
*   **New Mechanism:** The loop now acts like a digital circuit. It effectively runs `Accumulator = Adder(Accumulator, Term)` with discrete inputs.
*   **Prediction:** Train Acc > 90%. Extrapolation > 50%.
