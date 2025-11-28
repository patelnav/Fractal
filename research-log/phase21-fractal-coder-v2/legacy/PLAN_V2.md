# Plan: Fractal Coder v2 (Real-World Scaling)

## Goal
Scale the validated "Fractal Coder" architecture from synthetic math to real Python programming using `Qwen-2.5-Coder`.

## Architecture Mapping

| Component | Prototype (v1) | Real-World (v2) |
|:----------|:---------------|:----------------|
| **Generator** | Tiny 4-layer Transformer | `Qwen-2.5-Coder-7B-Instruct` (vLLM) |
| **Domain** | `ADD/SUB` Ops | HumanEval / MBPP (Python Functions) |
| **Roots** | Op Types | Function Signatures / Docstrings / High-Level Logic Blocks |
| **Execution** | `eval()` math | `exec()` Python unit tests (Sandboxed) |
| **Critic** | 2-Head Transformer | Learned Encoder (BERT/RoBERTa) or Finetuned Qwen |
| **Repair** | Mutate Root ID | In-Fill / Edit Prompting |

## Step 1: The "Fractal" View of Code
Standard LLMs generate tokens linearly. To apply the Fractal Loop, we need to view code hierarchically.
*   **Root:** The Function Signature + Docstring (The "Spec").
*   **Chunk:** The Implementation (The "Body").
*   *Simplification for v2:* We will treat the **Spec** as the frozen condition and the **Body** as the "Root" to be generated/repaired. Wait, that's just "Retry".
*   *True Fractal v2:* Break the body into logical blocks (e.g., "Init", "Loop", "Return").
    *   *Challenge:* Existing models aren't trained to generate chunks independently.
    *   *Solution:* Use **In-Filling** or **Structured Generation**.
    *   *Simplest Path:* Treat the entire function body as one "Root" for now, OR generate multiple candidate bodies (Flash Flood) and pick the best.
    *   *Better Path (Vector 7 style):* Use a **Sketch-then-Fill** approach.
        1.  Manager generates a "Sketch" (Comments/Pseudocode).
        2.  Renderer fills in the code for each comment in parallel.

## Proposed Pipeline (Sketch-Driven)
1.  **Sketch (Manager):** `Qwen` generates a plan:
    ```python
    # 1. Initialize map
    # 2. Iterate through input list
    # 3. Update counts
    # 4. Return max key
    ```
2.  **Flood (Renderer):** `Qwen` (Parallel Batch) expands each comment into code lines.
3.  **Execute:** Run tests.
4.  **Critic:** Identify which *Step* (1-4) failed (e.g., "Loop logic is wrong").
5.  **Patch:** Regenerate only Step 2.

## Implementation Steps
1.  **Environment:** Setup `vLLM` with `Qwen-2.5-Coder-7B` (or 1.5B for dev).
2.  **Dataset:** `HumanEval` (OpenAI).
3.  **Sketcher:** Prompt Qwen to generate "Step-by-step comments".
4.  **Renderer:** Prompt Qwen to "Write code for this comment".
5.  **Critic Data:** Generate failures, identify which block caused the crash/wrong output.
6.  **Loop:** Implement the `Generate -> Test -> Patch` cycle.

## Files
-   `research-log/phase21-fractal-coder-v2/design_v2.md`
-   `research-log/phase21-fractal-coder-v2/qwen_interface.py` (vLLM wrapper)
-   `research-log/phase21-fractal-coder-v2/humaneval_harness.py`
-   `research-log/phase21-fractal-coder-v2/fractal_loop_v2.py`
