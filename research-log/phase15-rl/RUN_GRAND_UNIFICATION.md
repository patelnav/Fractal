
# Phase 15.3: Grand Unification Experiment

## 1. Sync
```bash
python lambda_helper.py sync research-log/phase15-rl
```

## 2. Generate (GRPO Policy)
Generate 50 samples using the GRPO-tuned model.
```bash
ssh lambda
cd ~/Fractal/research-log/phase15-rl
source ../phase14-vector6-reboot/.venv/bin/activate

python generate_grpo_test.py
```

## 3. Execute (Labeling)
```bash
python execute_grpo_test.py
```

## 4. Rank (Critic Policy)
Use Phase 14 Critic to select the best code from Phase 15 Generator.
```bash
python test_grand_unification.py
```
