
# Phase 14.7: HumanEval Transfer Test

## 1. Sync
```bash
python lambda_helper.py sync research-log/phase14-vector6-reboot
```

## 2. Generate HumanEval (Remote)
Generate 50 samples per problem for the HumanEval benchmark.
```bash
ssh lambda
cd ~/Fractal/research-log/phase14-vector6-reboot
source .venv/bin/activate

python generate_humaneval.py
```

## 3. Execute & Label (Remote)
Run the generated code against HumanEval tests.
```bash
python execute_humaneval.py
```

## 4. Evaluate Transfer (Remote)
Use the **MBPP-trained Critic** to rank HumanEval solutions.
Does it generalize?

```bash
python test_humaneval_transfer.py
```
