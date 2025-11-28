# Phase 21 Report: Fractal Coder v2 -> v3

## Objective
Scale the Fractal Coder architecture from synthetic tasks to real-world Python programming (HumanEval) using `Qwen-2.5-Coder-7B`.

## Experiments & Results

We tested three architectural variants on a representative subset (15 hard problems) of HumanEval.

| Architecture | Approach | Pass Rate (15) | Diagnosis |
|:---|:---|:---|:---|
| **Baseline** | Prompt $\to$ Code | **66.7%** | Strong base performance. |
| **v2 (Flat)** | Sketch $\to$ Steps $\to$ Snippets $\to$ Assemble | 20.0% | **Failed.** Linear assembly breaks Python nesting/indentation. |
| **v2.5 (Tree)** | Sketch $\to$ Tree $\to$ Recursive Render | 0.0% | **Failed.** Models generate full code blocks, not atomic snippets. Assembly impossible. |
| **v3 (Guided)** | Sketch $\to$ "Plan: [Sketch]" $\to$ Code | **66.7%** | **Success.** Matches baseline. Solves assembly by delegating structure to the model. |

## Key Findings

1.  **The Assembly Trap:** Trying to "stitch" code snippets generated independently is a dead end for current LLMs. They are trained on contiguous text and lack the fine-grained control to output "just the inside of the loop".
2.  **Planning Works:** The v3 approach (generating a plan first) does not degrade performance. This is critical because the **Plan is the interface for Repair**.
3.  **Output Pollution:** 80% of v3 failures were due to the model executing the code or adding prose after the function. This is a solvable "harness" issue.

## Architecture for Phase 22 (Repair Loop)

We will proceed with **Fractal Coder v3**.

*   **Generator:** `Sketch -> Code` (Guided Mode).
*   **Critic:** Executes tests. If fail -> Identify failing test case.
*   **Repair:**
    1.  Feed `Sketch + Code + Error`.
    2.  Ask model to **Update Sketch** (Plan Repair).
    3.  Regenerate Code from New Sketch.

This avoids the "surgical edit" problem of v2 (patching lines) and treats the *Function Body* as the atomic unit of generation, with the *Sketch* as the atomic unit of reasoning.

## Artifacts
- `fractal_guided_v3.py`: The winning implementation.
- `qwen_interface.py`: vLLM wrapper.
- `humaneval_harness.py`: Execution sandbox.
