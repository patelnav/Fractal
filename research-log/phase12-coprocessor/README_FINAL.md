# Neural ALU: The Differentiable CPU

**Status:** SUCCESS (Phase 12.5 Verified)
**Date:** November 2025

## Overview
This directory contains the implementation of a **Neural Arithmetic Logic Unit (ALU)** capable of:
1.  **Addition:** Extrapolating to arbitrary bit-widths (trained on 8-bit, tested on 12-bit+).
2.  **Multiplication:** Zero-shot composition of the Adder into a Shift-and-Add loop.
3.  **Digital Stability:** Using "Digital Restoration" (Argmax/Gumbel-Softmax) to prevent analog drift in recursive loops.

## Key Components

### 1. The Fractal Adder (`model_logic.py` in Phase 11)
*   **Architecture:** Recurrent Transformer with GRU Gating.
*   **Insight:** Sharing weights across time (recurrence) forces the model to learn the *algorithm* of carry propagation rather than memorizing the addition table.
*   **Performance:** 99.5% Accuracy on Extrapolation (OOD).

### 2. The Fractal Multiplier (`model_fractal_mult.py`)
*   **Architecture:** A hard-coded "Shift-and-Add" loop that calls the frozen Adder.
*   **Insight:** **Digital Restoration**.
    *   Problem: Passing soft embeddings through a loop $O(N^2)$ deep causes signal degradation (noise).
    *   Solution: At every step, snap the accumulator logits to discrete indices (Argmax) and re-embed them. This creates a "Digital Repeater" effect, blocking noise.
*   **Performance:** 100% Accuracy on 8-bit Extrapolation (Zero-Shot).

## Reproduction

### 1. Verify the Primitives (Phase 12.5)
Run the diagnostic suite to verify the Adder, Shifter, and Loop logic:
```bash
python debug_phase12_wiring.py
```
Expected Result: `ALL SYSTEM CHECKS PASSED`.

### 2. Run the Zero-Shot Extrapolation Test
```bash
python test_mult_digital.py
```
Expected Result: `Final Accuracy: 100.00%`.

## Artifacts
*   `checkpoints/`: Contains the frozen Adder weights (symlinked from Phase 11).
*   `model_fractal_mult.py`: The reference implementation of the Neural CPU loop.
