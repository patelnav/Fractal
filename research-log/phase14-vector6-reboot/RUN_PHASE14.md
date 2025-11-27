
# Phase 14: Vector 6 Reboot - Execution Plan

## 1. Sync
```bash
python lambda_helper.py sync research-log/phase14-vector6-reboot
```

## 2. Generate Data (Remote)
We use `Qwen2.5-Coder-1.5B-Instruct` with **vLLM** for high-throughput generation.

```bash
ssh lambda
cd ~/Fractal/research-log/phase14-vector6-reboot
source .venv/bin/activate

# Install dependencies
pip install vllm datasets transformers

python generate_mbpp.py
```

## 3. Execute & Label (Remote)
Run the generated code against the unit tests to create the ground truth labels.
```bash
python execute_mbpp.py
```

## 4. Train Critic (Remote)
Train the Soft Verifier (Qwen-1.5B + Head) on the labeled data.
```bash
python train_critic.py
```

## 5. Success Criteria
*   **Critic Validation Accuracy > 80%**: Indicates the model can distinguish correct/incorrect code.
*   **Extrapolation Test (Phase 14.5)**: We will run a separate script to test if ranking by Critic improves Pass@1 on the Test Set.
