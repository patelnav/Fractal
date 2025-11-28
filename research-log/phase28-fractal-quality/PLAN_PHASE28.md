# Phase 28: Fractal Quality (Jacobi Decoding Proof)

## Objective
Prove that **Parallel Decoding** (Flash Flood) can produce **High Quality Output**, not just speed.
We will implement a **Jacobi Decoding** loop on top of `Qwen-2.5-1.5B-Instruct`.

## The Hypothesis
"A standard Causal LLM can settle into a coherent high-probability sequence from a rough initialization (Sketch) in parallel steps."

## Experiment Design
1.  **Model:** `Qwen/Qwen2.5-1.5B-Instruct`.
2.  **Prompt:** "Write a short Python function to calculate Fibonacci numbers."
3.  **Initialization (The 'Sketch'):**
    -   We will simulate a "Sketch" by taking the Ground Truth (AR generated) and masking 50% of tokens (or adding noise).
    -   Or, strictly: Initialize with the *previous* token repeated (Naive Jacobi).
4.  **Algorithm:**
    -   Input: `[Prompt, M_1, M_2, ..., M_N]` (Masks/Guesses).
    -   Loop 10 times:
        -   logits = `model([Prompt, Current_Guess])`
        -   `Next_Guess[i]` = `argmax(logits[i-1])`
    -   Check convergence.
5.  **Metric:**
    -   **Token Match Rate:** % of tokens matching the standard Greedy AR output.
    -   **N-Gram Accuracy:** Is the text coherent?

## Implementation
`jacobi_demo.py`:
-   Load Model.
-   Run Standard AR -> `Truth`.
-   Run Jacobi Loop (Init = Pad tokens).
-   Print intermediate states to visualize the "Wavefront" of convergence.

## Why this matters
If this works, it confirms that Phase 26/27's speed can be coupled with Phase 18/22's sketches to produce **Correct** code 10x faster.
