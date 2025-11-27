
# Running Phase 12.5: Wiring Debug

## 1. Sync
```bash
python lambda_helper.py sync research-log/phase12-coprocessor
```

## 2. Run Diagnostics (Remote)
We are running unit tests on the Frozen Adder and the Shift logic to ensure the components adhere to the contract required for multiplication.

```bash
ssh lambda
cd ~/Fractal/research-log/phase12-coprocessor
source .venv/bin/activate

python debug_phase12_wiring.py
```

## 3. Interpret Results
*   **Test 1 (Shift):** If this fails, our `shift_embedding` logic is wrong (e.g., padding with wrong vector).
*   **Test 2 (Zero):** If this fails, the Adder assumes specific padding or positional embeddings that we are violating.
*   **Test 3 (Accum):** If this fails, the Adder cannot handle sums of shifted inputs.
*   **Test 4 (Mult):** If this fails but 1-3 pass, the loop logic (digital restoration) is accumulating errors.
