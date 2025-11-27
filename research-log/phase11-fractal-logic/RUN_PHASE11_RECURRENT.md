
# Running Phase 11.1: Recurrent Fractal Logic

## 1. Sync Code
```bash
python lambda_helper.py sync research-log/phase11-fractal-logic
```

## 2. Run Training (Remote)
```bash
ssh lambda
cd ~/Fractal/research-log/phase11-fractal-logic
source .venv/bin/activate

# Train the Recurrent Model
python train_logic.py
```

## 3. Hypothesis
*   **Previous Model (Control):** Failed extrapolation (0% acc) because it learned position-specific rules.
*   **New Model (Recurrent):** Should succeed ($>90\%$ acc) because it uses the **same** weights for every bit position. It must learn the generic "Full Adder" logic to minimize loss on the training set.

## 4. Clean Up
```bash
# When done
python lambda_helper.py terminate
```
