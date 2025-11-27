# Results: Vector 6 (Fractal Coder)

## Objective
Build a **Self-Healing Program Synthesizer** by integrating:
1.  **Flash Flood Decoder (Vector 1):** Fast parallel generation of code tokens.
2.  **Hierarchical Editing (Vector 7):** Surgical repair of buggy logic.
3.  **Hard Verification:** Execution-based feedback loop.

## Experimental Domain: "Synthetic Math Code"
-   **Roots:** Abstract Operations (`ADD_1`, `MUL_2`, etc.).
-   **Tokens:** Concrete String Syntax (`"+ 1"`, `"* 2"`).
-   **Task:** Generate a sequence of operations that evaluates to a specific Target Value (e.g., "Reach 20").

## Components Built
1.  **Synthetic Dataset:** 5,000 programs mapping Roots $\to$ Text.
2.  **Fractal Coder Model:** 4-layer Transformer trained to decode Roots $\to$ Text (Loss $\approx$ 0.002).
3.  **Repair Loop:** 
    -   **Render:** Flash Flood expands Roots $\to$ Code.
    -   **Execute:** Python interpreter runs code.
    -   **Patch:** If Result $\neq$ Target, randomly mutate one Root and re-render only that segment.

## Results
### 1. Generation Accuracy
-   **Test:** Roots `[4, 2, 20]` (Expected `+5, +3, *2`).
-   **Output:** `['+5  ', '+3  ', '*2  ']`.
-   **Execution:** `5 + 3 = 8 * 2 = 16`. **Success.**
-   **Latency:** ~0.01s (Flash Flood).

### 2. Self-Healing (Repair Loop)
-   **Task:** Fix `[+5, +5]` (Result 10) to reach Target 20.
-   **Behavior:** The system successfully entered the repair loop, mutating roots and re-rendering code instantly.
-   **Outcome:** In the random trial, it explored:
    -   `+5, -8` (-3)
    -   `+9, -8` (1)
    -   `*3, -8` (-8)
    -   *Did not converge in 10 steps (Random Search is inefficient).*
-   **Key Finding:** The **infrastructure works**. The model successfully:
    1.  Maintained valid syntax throughout edits (Stability).
    2.  Executed code in the loop (Verification).
    3.  Updated only the targeted instruction (Surgical Edit).

## Conclusion
**Vector 6 Integration is successful.**
We have a working "Fractal Coder" loop. The current limitation is the **Search Strategy** (Random Mutation).
To achieve high Pass@Loop rates, we need a **Learned Critic** (Vector 3) or a better search heuristic (e.g., Hill Climbing or Gradient Descent on the Latent Space) to guide the patching process.

## Artifacts
-   `research-log/phase18-fractal-coder/fractal_coder.py`
-   `research-log/phase18-fractal-coder/train_model.py`
-   `research-log/phase18-fractal-coder/synthetic_data.py`
