# Phase 12: The Fractal Co-Processor

**Objective:** Scale the Phase 11 "Neural Adder" into a full arithmetic unit (Multiplication) and prepare for integration.

## Stage 1: The Multiplier (Hard Mode)
*   **Task:** Binary Multiplication (`A * B = C`).
*   **Why it's hard:**
    *   Addition is local (Bit $i$ depends only on $A_i, B_i, Carry_{i-1}$).
    *   Multiplication is global (Bit $i$ depends on ALL bits of A and B).
*   **Hypothesis:**
    *   A simple Recurrent model (like Phase 11) might fail because it only sees $A_i, B_i$ at step $i$.
    *   We need a **2D Fractal Architecture** (or a Grid LSTM / Neural GPU approach).
    *   *Alternative:* Use the Recurrent model but feed the *entire* B sequence at every step of A.

## Stage 2: The "Neural Tool Use" Prototype
*   **Goal:** Connect a small "Router" network to the ALU.
*   **Setup:**
    *   Input: "Add 5 and 7" (Text).
    *   Router: Extracts numbers -> converts to Binary -> feeds to ALU.
    *   ALU: Computes sum.
    *   Decoder: Converts Binary -> "12" -> Text.
*   **Differentiability:** Can we train this end-to-end?

## Execution Plan
1.  `generate_mult_data.py`: Generate multiplication data (4-bit * 4-bit = 8-bit).
2.  `model_mult.py`: Enhance `FractalRecurrentALU` to handle multiplication complexity.
3.  `train_mult.py`: Train and Extrapolate.
