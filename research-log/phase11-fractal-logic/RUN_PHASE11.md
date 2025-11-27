
# Running Phase 11: Fractal Logic on Lambda

## 1. Setup
```bash
# Launch Instance (if not running)
python lambda_helper.py launch gpu_1x_a100
python lambda_helper.py wait
python lambda_helper.py setup-ssh

# Sync Code
python lambda_helper.py sync research-log/phase11-fractal-logic
```

## 2. Execution (On Remote)
```bash
ssh lambda

# Navigate
cd ~/Fractal/research-log/phase11-fractal-logic

# Install Dependencies (Minimal)
# We don't strictly need 'uv' for this simple script, standard pip is fine, 
# but sticking to 'uv' is good practice.
pip install uv
~/.local/bin/uv venv --python 3.12 .venv
source .venv/bin/activate
~/.local/bin/uv pip install torch tqdm

# Generate Data (if not synced or to be safe)
python generate_binary_data.py

# Run Training
python train_logic.py
```

## 3. Monitor
*   Watch the "Test Acc (12-bit Extrapolation)" metric.
*   If it stays at 0.0, the model is memorizing.
*   If it climbs, we have achieved **Generalization**.

## 4. Cleanup
```bash
# Local
python lambda_helper.py terminate
```
