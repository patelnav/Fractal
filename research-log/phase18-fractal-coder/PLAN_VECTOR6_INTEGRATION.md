# Plan: Vector 6 "Fractal Coder" (Integration)

## Goal
Combine **Flash Flood (Vector 1)** and **Hierarchical Editing (Vector 7)** to build a high-performance, self-healing Program Synthesizer.

## The Thesis
1.  **Speed:** Use Flash Flood to generate $K$ candidate solutions in parallel ($O(1)$ render time).
2.  **Verification:** Execute candidates against unit tests (Hard Verification).
3.  **Stability:** If a candidate fails, identify the specific "Chunk/Root" responsible (e.g., a buggy function) and **Surgically Edit** it (Vector 7) while preserving the correct parts.

## Architecture: The "Fractal Repair Loop"
1.  **Draft:** Manager generates a program skeleton (Roots).
2.  **Flood:** Flash Flood renders the code (Tokens).
3.  **Verify:** Run unit tests.
    - *Pass:* Return code.
    - *Fail:* Identify failing test case $\to$ Localize error to a Root (heuristically or via attention attribution).
4.  **Patch:** 
    - Keep good Roots locked.
    - Sample new candidates for the bad Root (Best-of-K Local Flood).
    - Re-render only that segment.
5.  **Loop:** Repeat until pass or timeout.

## Step 1: Domain Adaptation (Toy Python)
Since our current model is trained on Shakespeare (Char-level), it cannot write real Python. 
For this phase, we will simulate the "Code Domain" using a **Synthetic Task**:
- **Task:** Generate a list of mathematical expressions that sum to a target.
- **Roots:** Operations (e.g., `ADD`, `SUB`, `MUL`).
- **Chunks/Text:** The string representation (e.g., `"+ 5"`, `"* 2"`).
- **Verification:** Evaluate the string and check if it equals the target.

*Note: We are testing the **System Architecture** (Manager + Flash Flood + Editor + Verifier), not the underlying LLM's knowledge of Python syntax (which we proved in Phase 14 with Qwen).*

## Implementation Plan
1.  **Synthetic Dataset (`synthetic_code.py`):
    - Generate "Programs" (sequences of ops).
    - Tokenizer: Roots = {`OP_ADD`, `OP_SUB`, ...}, Chunks = {`+`, `-`, digits}.
2.  **Train Mini-Model (`train_fractal_coder.py`):
    - Train a small Fractal Engine on this synthetic data.
3.  **The Fractal Coder (`fractal_coder.py`):
    - Implements `generate_and_repair(target_value)`.
    - Uses Flash Flood for initial generation.
    - Uses Vector 7 patching for repair.
4.  **Evaluation (`test_fractal_coder.py`):
    - Compare Pass@1 (No Repair) vs Pass@Loop (With Repair).

## Files
- `research-log/phase18-fractal-coder/synthetic_data.py`
- `research-log/phase18-fractal-coder/train_model.py`
- `research-log/phase18-fractal-coder/fractal_coder.py`
- `research-log/phase18-fractal-coder/RESULTS.md`
