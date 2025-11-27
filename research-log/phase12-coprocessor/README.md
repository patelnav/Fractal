# Phase 12: The Fractal Multiplier (Composing Logic)

**Objective:** Build a Differentiable Multiplier by *recursively* using the Phase 11 Adder.

## The Theory: "Logic is Fractal"
*   **Level 0 (Bit):** NAND gate.
*   **Level 1 (Linear):** Addition is a sequence of Bit operations (Carry propagation).
*   **Level 2 (Quadratic):** Multiplication is a sequence of **Additions**.

Instead of training a black-box Transformer to "guess" the product, we construct a **Recursive Architecture** that mirrors this mathematical truth.

## The Architecture: Shift-and-Add
The model consists of two parts:
1.  **The Core (The Adder):** The `FractalRecurrentALU` from Phase 11. It knows how to add two arbitrary binary strings.
2.  **The Controller (The Looper):** A new module that scans the multiplier $B$.
    *   For each bit $b_i$ in $B$:
        *   It prepares a term $T_i = (A \ll i)$ if $b_i=1$, else $0$.
        *   It calls the **Adder** to update the running total: $Sum = Adder(Sum, T_i)$.

## Why this is "Fractal"
It is **Self-Similar**: The Multiplier *contains* the Adder. If we wanted to do Exponentiation, we would build a module that *contains* the Multiplier.
This **Compositionality** allows us to solve tasks of infinite complexity by stacking simple, solved primitives.

## Differentiability
Because the **Adder** (Phase 11) is fully differentiable, we can backpropagate through the entire loop. The Controller learns *how* to use the Adder to perform multiplication.
