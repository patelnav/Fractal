
# Running Phase 14.5: Extrapolation Test

## 1. Sync
```bash
python lambda_helper.py sync research-log/phase14-vector6-reboot
```

## 2. Generate Test Data (Remote)
We generate 50 samples for each problem in the **Test Split**.
```bash
ssh lambda
cd ~/Fractal/research-log/phase14-vector6-reboot
source .venv/bin/activate

python generate_mbpp_test.py
```

## 3. Execute & Label (Remote)
Get ground truth for evaluation.
```bash
python execute_mbpp_test.py
```

## 4. Evaluate Critic Ranking (Remote)
Score all test samples with the Phase 14 Critic (Epoch 1).
Compare Random Selection vs. Critic Selection.

```bash
python test_critic_ranking.py
```

## 5. Success Criteria
*   **Critic Pass@1 > Baseline Pass@1**: Proves the verifier extrapolates.
*   **Target**: +5% to +10% improvement.
