
# Running Phase 13: Curriculum Learning

## 1. Sync
```bash
python lambda_helper.py sync research-log/phase13-neural-riscv
```

## 2. Generate Traces (Remote)
We generate "Teacher Traces" (State -> Action sequences) for the standard Shift-and-Add multiplication algorithm.
```bash
ssh lambda
cd ~/Fractal/research-log/phase13-neural-riscv
source .venv/bin/activate

python generate_trace_data.py
```

## 3. Train via Imitation (Remote)
We train the Neural Controller to mimic the traces.
Because the Controller now sees the full register bits (FIXED from previous attempt), it should be able to learn the conditional logic:
"If bit `t` of Register B is 1, then `ADD`, else `NOOP`."

```bash
python train_curriculum.py
```

## 4. Success Criteria
*   **Op Accuracy > 99%**: The model learns the program.
*   This proves the Neural RISC-V architecture is *capable* of holding the algorithm, paving the way for future RL/Search approaches to *find* it without traces.
