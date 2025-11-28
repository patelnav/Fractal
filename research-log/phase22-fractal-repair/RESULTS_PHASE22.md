# Phase 22 Report: Fractal Repair Loop

## Objective
Validate the "Fractal Repair" hypothesis: that iteratively fixing the **Sketch (Plan)** based on execution feedback is more effective than patching code or regenerating blindly.

## Method
- **Base Model:** Qwen-2.5-Coder-7B (via vLLM).
- **Architecture:** v3 Sketch-Guided (Sketch $\to$ Code).
- **Loop:**
    1. Generate Sketch & Code.
    2. Run HumanEval Tests (Oracle).
    3. If Fail: Prompt model with `(Sketch, Code, ErrorTrace)`. Ask for `Revised Sketch`.
    4. Regenerate Code from `Revised Sketch`.
    5. Max 5 Retries.
- **Dataset:** Representative Set (15 hard problems).

## Results

| Metric | Value |
|:---|:---|
| **Baseline (Pass@1)** | 66.7% (10/15) |
| **Repair (Pass@Repair)** | 66.7% (10/15) |
| **Repair Success Rate** | **0%** (0/5 failed problems fixed) |
| **Avg Iterations (Failures)** | 5.0 (Maxed out) |

## Failure Analysis
The 5 failing problems (HumanEval/10, 32, 33, 39, 129) resisted all repair attempts.

### Hypotheses for Failure
1.  **The "Hard" Barrier:** The failing problems involve specific algorithmic insights (e.g., Grid Search in #129, Math in #39) that the model simply lacks, regardless of planning. "Planning harder" doesn't invent missing knowledge.
2.  **Feedback Disconnect:** The error trace (often `AssertionError`) usually doesn't point to the *flaw in the plan*. It just says "Wrong Answer". The model guesses at a fix but often thrashes (No-Op repairs or regressions).
3.  **Abstraction Gap:** Fixing a "Plan" is great for structural errors (nesting, flow), but bad for off-by-one errors or edge cases. A high-level plan says "Iterate array", but the bug is `range(len(arr)-1)`. The plan doesn't capture this detail, so updating the plan doesn't fix the code.

## Conclusion
**Plan-Level Repair is not a silver bullet for algorithmic logic bugs.**
While it stabilizes generation (v3), it lacks the granularity to fix specific implementation details driven by test failures.

## Next Steps
We need to bridge the gap between "High Level Plan" and "Low Level Bug".
- **Option A:** **Critic-Guided Code Repair.** (Vector 3). Don't touch the plan. Use a Critic to pinpoint the *line* of code to fix, then patch.
- **Option B:** **Sampling/Diversity.** If the model is stuck, generating 5 *different* plans might be better than refining one.
