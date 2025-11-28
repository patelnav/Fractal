# Phase 22: Fractal Repair Loop

## Objective
Turn the **Fractal Coder v3** (Sketch-Guided) into an iterative problem solver by implementing a **Plan-Level Repair Loop**.

## Core Hypothesis
"It is easier to fix a high-level thought (Sketch) than to patch low-level implementation details."

## Architecture

### 1. The Primitive (v3 Guided)
- **Input:** Signature, Docstring, Sketch
- **Process:** Sketch -> Single-Shot Guided Generation
- **Output:** Complete Function Body

### 2. The Loop
1.  **Initial Generation:** Generate `Sketch_0` and `Code_0`.
2.  **Execute:** Run tests (Oracle/HumanEval).
    - If **Pass**: Return Success.
    - If **Fail**: Capture `Error_Trace`.
3.  **Repair Decision:**
    - Input: `Signature`, `Docstring`, `Sketch_t`, `Code_t`, `Error_Trace`.
    - Prompt: "Here is the plan, code, and error. Rewrite the plan to fix the logic."
    - Output: `Sketch_{t+1}`.
4.  **Regenerate:** Generate `Code_{t+1}` from `Sketch_{t+1}`.
5.  **Repeat** until Success or Max Iterations (K=5).

## Implementation Details

### Prompt Engineering (Repair)
```text
The following Python implementation failed tests.
Signature: ...
Docstring: ...

Plan Used:
...

Code Generated:
...

Error Trace:
...

Analyze the error. Then, provide a REVISED implementation plan (Sketch) that fixes the logic.
Do not write code yet. Just the corrected plan.
```

### Control Logic
- **Max Iterations:** 5
- **No-Op Detection:** If `Sketch_{t+1} == Sketch_t` (or very similar), abort or increase temperature.
- **Logging:** Save the full trajectory `[Sketch_0, Code_0, Error_0, Sketch_1, ...]`.

## Evaluation Metric
Compare:
1.  **Baseline (Pass@1):** Success rate at $t=0$.
2.  **Repair (Pass@Repair):** Success rate within $K$ iterations.
3.  **Delta:** `Pass@Repair - Pass@1`.
4.  **Cost:** Average tokens per problem.

## Tasks
- [ ] Copy/Update `humaneval_harness.py` to return detailed error traces.
- [ ] Extend `QwenInterface` with `repair_sketch` method.
- [ ] Implement `fractal_repair.py` (The Loop).
- [ ] Run on Representative Set (15 problems).
