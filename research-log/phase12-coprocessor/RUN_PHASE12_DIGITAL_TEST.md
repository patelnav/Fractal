
# Running Phase 12.3: Zero-Shot Digital Restoration

## 1. Sync
```bash
python lambda_helper.py sync research-log/phase12-coprocessor
```

## 2. Run Test (Remote)
We have frozen the Adder weights and implemented **Digital Snapping** (Argmax) in the loop.
This tests if a pure Neural Adder can be composed algorithmically without training.

```bash
ssh lambda
cd ~/Fractal/research-log/phase12-coprocessor
source .venv/bin/activate

# Ensure using the GRU-Gated checkpoint from Phase 11
# (If you re-ran Phase 11, check ckpt_e10.pt exists)

python test_mult_digital.py
```

## 3. Success Criteria
*   If Acc > 90%, we have proven **Fractal Compositionality**.
*   This means we can build complex algorithms (Exp, Modulo, RSA) by just stacking pre-trained primitives, without ever training the complex task end-to-end.
