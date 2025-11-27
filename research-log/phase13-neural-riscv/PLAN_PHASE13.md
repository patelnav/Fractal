# Phase 13: The Neural RISC-V

**Objective:** Generalize the "Hard-Coded Loop" from Phase 12 into a **Learned Controller**.

## The Problem
Phase 12 proved that *if* we arrange the Adder in a Shift-and-Add loop, it works perfectly.
But we don't want to hard-code every algorithm (Division, Exponentiation, etc.).
We want the AI to *invent* these algorithms.

## The Solution: Neural ISA
We treat the Phase 11 Adder as a **Differentiable Instruction**.

**Components:**
1.  **Registers:** A bank of embedding vectors (e.g., `R0`...`R4`).
    *   `R0`, `R1` initialized with Input A, B.
2.  **ALU:** Our frozen `FractalRecurrentALU`.
3.  **Controller:** A small Transformer that outputs a sequence of instructions.
    *   `Action_t = Policy(Registers_t)`
    *   Example: `R2 = ADD(R0, R1)`

## Experiment 13.0: Learning Multiplication
*   **Task:** Train the Controller to output `C = A * B`.
*   **Constraint:** It has NO multiplication head. It ONLY has the `ADD` instruction and a `SHIFT` instruction.
*   **Goal:** The Controller must learn to output the sequence:
    *   `SHIFT R0`
    *   `ADD R2, R0`
    *   ... (The Shift-and-Add algorithm).

**Why this is huge:**
If successful, we are effectively doing **Program Synthesis** but in **Embedding Space**. The model isn't writing Python code; it's writing "Neural Assembly".
